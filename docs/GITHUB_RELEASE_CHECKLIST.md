# GitHub release checklist

The manuscript has no appendix; this repository must contain all technical
material referenced as supplementary implementation detail.

- [x] Record the recovered EasyJailbreak and PAIR commits while retaining the source-hash lock.
- [ ] Confirm `docs/PAPER_REPOSITORY_MAP.md` matches the final manuscript numbering.
- [ ] Confirm the paper's Data/Code Availability paragraph points to the final repository release/DOI.
- [ ] Confirm no manuscript appendix is referenced as a separate uploaded file.
- [ ] Confirm the repository visibility and responsible-use statement with the supervisor.
- [ ] Upload the contents of this directory at the repository root, not the outer ZIP directory.
- [ ] Enable GitHub Actions and confirm `public-release-checks` passes.
- [ ] Create a `v1.0.4` tag only after the manifest is unchanged.
- [ ] Attach the release ZIP and its SHA-256 sidecar to the GitHub Release page.
- [ ] Do not upload private raw-output archives, `.env`, logs, model checkpoints, or caches.
- [ ] Confirm inactive snapshots exist only under `archive/` and no active module imports them.
- [ ] Run `make manifest`, then `make verify-release`.
