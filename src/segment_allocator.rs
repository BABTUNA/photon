//! mmap-backed segment allocator.
//!
//! Creates a fixed-size file on disk, maps it into the process address space,
//! and hands out page-aligned offsets via a lock-free bump pointer. No
//! deallocation yet — eviction lands later in W3.

use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const PAGE_SIZE: usize = 4096;

/// Round `n` up to the next multiple of [`PAGE_SIZE`].
fn page_align_up(n: usize) -> usize {
    (n + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Fixed-size mmap region with a lock-free bump-pointer allocator.
///
/// Each call to [`allocate`] CAS-bumps an `AtomicUsize` offset, returning the
/// starting offset of a page-aligned range. The mmap itself is mutex-guarded
/// for now; finer-grained synchronization will arrive once we need zero-copy
/// concurrent reads.
pub(crate) struct SegmentAllocator {
    mmap: Mutex<MmapMut>,
    capacity: usize,
    next: AtomicUsize,
}

impl SegmentAllocator {
    /// Create or truncate the file at `path`, resize it to `capacity` bytes,
    /// and `mmap` it into the process.
    pub(crate) fn new(path: &Path, capacity: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(capacity as u64)?;
        // SAFETY: We own the file and no other process or thread holds a
        // mapping to it at this point. The mapping is valid for the
        // lifetime of `self.mmap`.
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self {
            mmap: Mutex::new(mmap),
            capacity,
            next: AtomicUsize::new(0),
        })
    }

    /// Reserve a contiguous, page-aligned region of at least `size` bytes.
    /// Returns the starting offset on success, or `None` if the segment is
    /// full.
    pub(crate) fn allocate(&self, size: usize) -> Option<usize> {
        let aligned = page_align_up(size);
        loop {
            let current = self.next.load(Ordering::Relaxed);
            let new_next = current.checked_add(aligned)?;
            if new_next > self.capacity {
                return None;
            }
            match self.next.compare_exchange_weak(
                current,
                new_next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(current),
                Err(_) => continue,
            }
        }
    }

    /// Copy `payload` into the segment starting at `offset`.
    ///
    /// The caller is responsible for ensuring `offset` came from a prior
    /// successful [`allocate`] call and `offset + payload.len() <= capacity`.
    pub(crate) fn write_at(&self, offset: usize, payload: &[u8]) {
        let mut mmap = self.mmap.lock().unwrap();
        mmap[offset..offset + payload.len()].copy_from_slice(payload);
    }

    /// Read `len` bytes from the segment starting at `offset`.
    ///
    /// Returns a freshly-allocated `Vec`; zero-copy reads will be possible
    /// once the mutex is replaced with finer-grained synchronization.
    pub(crate) fn read_at(&self, offset: usize, len: usize) -> Vec<u8> {
        let mmap = self.mmap.lock().unwrap();
        mmap[offset..offset + len].to_vec()
    }

    /// Total bytes mapped, including unused tail.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}
