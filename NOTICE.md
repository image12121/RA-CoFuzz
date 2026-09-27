# Notices

This release integrates or adapts code from upstream research repositories.

- `GPTFuzz-master/` retains its included license file and original-source hashes.
- `JailbreakingLLMs-official/` retains its included license file.
- `pair_official_autodl/` is an integration layer for the PAIR workflow; consult the associated upstream repository and paper when redistributing.
- EasyJailbreak is used as a runtime dependency by the compatibility layer but its package cache is not bundled.
- Inactive pre-fix RA-CoFuzz source snapshots are retained only under `archive/legacy_snapshots/` for provenance. They are not imported or executed.

The top-level MIT License applies to RA-CoFuzz release-specific code and documentation only. It does not replace third-party licenses. The bundled PAIR source retains upstream example paths such as `/home/pchao/...`; these are public upstream defaults, not paths from the RA-CoFuzz experiment host, and the RA-CoFuzz adapters do not rely on them. Before publishing, verify that all upstream notices required by the target GitHub release are present.
