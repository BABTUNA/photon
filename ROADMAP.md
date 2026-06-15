# photon roadmap

A 4-week plan to a minimal, distributed Python runtime (`v0.1`). photon
re-implements Ray's core abstractions — tasks, object refs, a shared object
store, a scheduler, and (stretch) actors — to understand how they fit together.

Schedule assumes the start of **Week 1 = 2026-06-14**; target `v0.1` by
**2026-07-11**.

## Done (foundation)

- PyO3 extension boundary; `@remote` decorator + `Future` stub.
- `.remote()` wired to Rust; tasks run on the Tokio blocking pool.
- cloudpickle arg marshaling; worker unpickles, runs, re-pickles the result.
- `ObjectStore` trait; content-addressed (blake3) object IDs.
- `SegmentAllocator`: mmap-backed, page-aligned, lock-free bump pointer.
- `MmapObjectStore`: payloads in the mmap segment, indexed by content hash;
  `put` is idempotent (dedup); `release` drops the index entry only.

## Week 1 (Jun 14–20) — ObjectRefs through the task path

- `.remote()` returns an `ObjectRef` (wraps an `ObjectId`) instead of blocking.
- Task results are pickled and `put` into the store; `photon.get(ref)` fetches
  and unpickles.
- ObjectRefs passed as task args are resolved before the function runs →
  **task chaining / dependencies**.
- `MmapObjectStore` becomes the default backing store; `InMemoryObjectStore`
  retires to test-only.
- **Done when:** `b = f.remote(a.remote())` works and `get(b)` is correct.

## Week 2 (Jun 21–27) — Eviction & memory management ("W3" in code comments)

- Reference counting for `ObjectId`s (live refs + pending-task holds).
- Free-list to replace the pure bump pointer, so `release` reclaims pages.
- LRU eviction when the segment is full; a real recovery path for `put → None`.
- **Done when:** a run that exceeds segment capacity keeps working by evicting
  dead objects.

## Week 3 (Jun 28–Jul 4) — Multi-process workers & scheduler (highest risk)

- Separate **worker processes** instead of in-process GIL threads.
- Unix-socket protocol: driver dispatches tasks, workers report completion.
- Workers map the **same segment file** → zero-copy-ish handoff (Plasma-style).
- Simple scheduler: task queue → dispatch to idle workers.
- **Done when:** tasks run in real subprocesses against the shared store.

## Week 4 (Jul 5–11) — Robustness & polish (actors = stretch)

- Error propagation: task exceptions re-raise on `.get()`.
- Zero-copy reads: drop the allocator mutex (the deferred comment item).
- Docs, a runnable example, a small benchmark, README update.
- **Stretch:** stateful **actors** — `@remote` classes pinned to a worker,
  methods returning ObjectRefs. Pulled in only if Weeks 1–3 land early.
- **Done when:** `v0.1` is tagged with docs + example + green tests.
