# Contributing

Contributions should preserve the frozen result release and keep experimental changes explicit.

1. Add new protocol variants under a new versioned configuration; do not overwrite `extensions_q20/protocol.json`.
2. Keep raw conversations and credentials outside the repository.
3. Add dependency-free regression tests for adapters and accounting logic.
4. Run `make verify-public`, `make verify-extensions`, and `make figures` before submitting a change.
5. Report aggregate results with all configured seeds and disclose early-stopping/query-count differences.

Generated figures should be reproducible from content-free aggregate data and supplied in PDF, SVG, and PNG.

