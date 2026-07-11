# mini-multibase

A distributed query engine in Rust, specialized for multimodal (video + sensor) data.

This repo previously hosted **photon**, a mini distributed Python runtime; that project lives in the git history prior to this commit.

## What's being built

- **Part 1:** a Rust replica of the query engine from [How Query Engines Work](https://howqueryengineswork.com/) — Arrow-based type system, logical/physical plans, DataFrame + SQL frontends, optimizer — plus a distributed layer: scheduler/workers over gRPC, an Arrow Flight data plane, hash-partitioned shuffle, and OTEL/Jaeger tracing.
- **Part 2:** specialization into a mini **MultiBase** — a multimodal table where video frames and sensor rows live timestamp-aligned in one dataset, semantic clip search in plain English (CLIP embeddings + distributed top-k), an ASOF join, and query results served over Arrow Flight into PyTorch tensors.

Status: repo reset — engine scaffolding lands next.
