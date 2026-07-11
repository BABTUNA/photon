//! mini-multibase query engine.
//!
//! Layers land in plan order: type system, data sources, logical plans,
//! physical plans, then distributed execution.

// Placeholder so the crate has something to compile and test until the
// first real module lands (commit 1.3).
pub const ENGINE_NAME: &str = "mini-multibase";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(super::ENGINE_NAME, "mini-multibase");
    }
}
