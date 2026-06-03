//! Object store: stores byte payloads by ID and retrieves them later.
//!
//! The trait defined here is the abstract contract. The in-memory impl is
//! a simple HashMap-backed baseline used to exercise the trait shape while
//! the mmap-backed implementation is being built out.

use std::collections::HashMap;
use std::sync::Mutex;

/// 32-byte object identifier — blake3 hash of the payload bytes.
///
/// Content addressing means two identical payloads share an ID; the store
/// dedupes implicitly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ObjectId(pub(crate) [u8; 32]);

/// Stores byte payloads by ID and retrieves them later.
pub(crate) trait ObjectStore {
    /// Insert a payload, returning its ID.
    fn put(&self, payload: Vec<u8>) -> ObjectId;

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
    fn put(&self, payload: Vec<u8>) -> ObjectId {
        let id = ObjectId(*blake3::hash(&payload).as_bytes());
        self.objects.lock().unwrap().insert(id, payload);
        id
    }

    fn get(&self, id: &ObjectId) -> Option<Vec<u8>> {
        self.objects.lock().unwrap().get(id).cloned()
    }

    fn release(&self, id: &ObjectId) {
        self.objects.lock().unwrap().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let store = InMemoryObjectStore::new();
        let id = store.put(vec![1, 2, 3]);
        assert_eq!(store.get(&id), Some(vec![1, 2, 3]));
    }

    #[test]
    fn release_makes_get_return_none() {
        let store = InMemoryObjectStore::new();
        let id = store.put(vec![1, 2, 3]);
        store.release(&id);
        assert_eq!(store.get(&id), None);
    }

    #[test]
    fn different_payloads_get_different_ids() {
        let store = InMemoryObjectStore::new();
        let id1 = store.put(vec![1]);
        let id2 = store.put(vec![2]);
        assert_ne!(id1, id2);
    }

    #[test]
    fn identical_payloads_get_identical_ids() {
        let store = InMemoryObjectStore::new();
        let id1 = store.put(vec![1, 2, 3]);
        let id2 = store.put(vec![1, 2, 3]);
        assert_eq!(id1, id2);
    }
}
