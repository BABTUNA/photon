//! Object store: stores byte payloads by ID and retrieves them later.
//!
//! The trait defined here is the abstract contract. The in-memory impl is
//! a simple HashMap-backed baseline used to exercise the trait shape while
//! the mmap-backed implementation is being built out.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Mutex;

use crate::segment_allocator::SegmentAllocator;

/// 32-byte object identifier — blake3 hash of the payload bytes.
///
/// Content addressing means two identical payloads share an ID; the store
/// dedupes implicitly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ObjectId(pub(crate) [u8; 32]);

/// Stores byte payloads by ID and retrieves them later.
pub(crate) trait ObjectStore {
    /// Insert a payload, returning its ID. Returns `None` if the store has no
    /// room left for the payload.
    fn put(&self, payload: Vec<u8>) -> Option<ObjectId>;

    /// Look up a payload by ID. Returns `None` if no such object exists.
    fn get(&self, id: &ObjectId) -> Option<Vec<u8>>;

    /// Drop a payload from the store.
    fn release(&self, id: &ObjectId);
}

/// In-memory baseline implementation of [`ObjectStore`].
///
/// Holds payloads in a HashMap keyed by their blake3 content hash. Useful as
/// a reference while the mmap-backed store is built; stays around for tests
/// against code that depends on the trait.
pub(crate) struct InMemoryObjectStore {
    objects: Mutex<HashMap<ObjectId, Vec<u8>>>,
}

impl InMemoryObjectStore {
    pub(crate) fn new() -> Self {
        Self {
            objects: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectStore for InMemoryObjectStore {
    fn put(&self, payload: Vec<u8>) -> Option<ObjectId> {
        let id = ObjectId(*blake3::hash(&payload).as_bytes());
        self.objects.lock().unwrap().insert(id, payload);
        Some(id)
    }

    fn get(&self, id: &ObjectId) -> Option<Vec<u8>> {
        self.objects.lock().unwrap().get(id).cloned()
    }

    fn release(&self, id: &ObjectId) {
        self.objects.lock().unwrap().remove(id);
    }
}

/// mmap-backed implementation of [`ObjectStore`].
///
/// Payload bytes live in a [`SegmentAllocator`]'s on-disk mmap region; an
/// in-memory index maps each content hash to the `(offset, len)` where its
/// bytes were written. Content addressing makes `put` idempotent: a payload
/// already in the index is not re-written.
///
/// `release` drops the index entry only — the segment space is not reclaimed
/// yet. Eviction lands in W3.
pub(crate) struct MmapObjectStore {
    segment: SegmentAllocator,
    index: Mutex<HashMap<ObjectId, (usize, usize)>>,
}

impl MmapObjectStore {
    /// Create a store backed by a fresh `capacity`-byte segment at `path`.
    pub(crate) fn new(path: &Path, capacity: usize) -> io::Result<Self> {
        Ok(Self {
            segment: SegmentAllocator::new(path, capacity)?,
            index: Mutex::new(HashMap::new()),
        })
    }
}

impl ObjectStore for MmapObjectStore {
    fn put(&self, payload: Vec<u8>) -> Option<ObjectId> {
        let id = ObjectId(*blake3::hash(&payload).as_bytes());
        let mut index = self.index.lock().unwrap();
        if index.contains_key(&id) {
            return Some(id);
        }
        let offset = self.segment.allocate(payload.len())?;
        self.segment.write_at(offset, &payload);
        index.insert(id, (offset, payload.len()));
        Some(id)
    }

    fn get(&self, id: &ObjectId) -> Option<Vec<u8>> {
        let (offset, len) = *self.index.lock().unwrap().get(id)?;
        Some(self.segment.read_at(offset, len))
    }

    fn release(&self, id: &ObjectId) {
        self.index.lock().unwrap().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let store = InMemoryObjectStore::new();
        let id = store.put(vec![1, 2, 3]).unwrap();
        assert_eq!(store.get(&id), Some(vec![1, 2, 3]));
    }

    #[test]
    fn release_makes_get_return_none() {
        let store = InMemoryObjectStore::new();
        let id = store.put(vec![1, 2, 3]).unwrap();
        store.release(&id);
        assert_eq!(store.get(&id), None);
    }

    #[test]
    fn different_payloads_get_different_ids() {
        let store = InMemoryObjectStore::new();
        let id1 = store.put(vec![1]).unwrap();
        let id2 = store.put(vec![2]).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn identical_payloads_get_identical_ids() {
        let store = InMemoryObjectStore::new();
        let id1 = store.put(vec![1, 2, 3]).unwrap();
        let id2 = store.put(vec![1, 2, 3]).unwrap();
        assert_eq!(id1, id2);
    }

    fn make_mmap_store(capacity: usize) -> (tempfile::TempDir, MmapObjectStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("segment");
        let store = MmapObjectStore::new(&path, capacity).unwrap();
        (dir, store)
    }

    #[test]
    fn mmap_put_get_round_trip() {
        let (_dir, store) = make_mmap_store(4096 * 4);
        let id = store.put(vec![1, 2, 3]).unwrap();
        assert_eq!(store.get(&id), Some(vec![1, 2, 3]));
    }

    #[test]
    fn mmap_get_unknown_id_returns_none() {
        let (_dir, store) = make_mmap_store(4096 * 4);
        let id = ObjectId([0u8; 32]);
        assert_eq!(store.get(&id), None);
    }

    #[test]
    fn mmap_release_makes_get_return_none() {
        let (_dir, store) = make_mmap_store(4096 * 4);
        let id = store.put(vec![1, 2, 3]).unwrap();
        store.release(&id);
        assert_eq!(store.get(&id), None);
    }

    #[test]
    fn mmap_identical_payloads_dedupe() {
        // Capacity for exactly one page; a second distinct payload would not
        // fit, so a successful second put proves the bytes were not re-written.
        let (_dir, store) = make_mmap_store(4096);
        let id1 = store.put(vec![1, 2, 3]).unwrap();
        let id2 = store.put(vec![1, 2, 3]).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.get(&id1), Some(vec![1, 2, 3]));
    }

    #[test]
    fn mmap_put_returns_none_when_full() {
        let (_dir, store) = make_mmap_store(4096);
        assert!(store.put(vec![1]).is_some());
        assert!(store.put(vec![2]).is_none());
    }
}
