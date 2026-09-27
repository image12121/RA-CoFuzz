# Security and responsible disclosure

Use RA-CoFuzz only for authorized model-safety evaluation. Do not point the runners at third-party services without explicit permission, and do not commit credentials, private endpoints, generated conversations, or model checkpoints.

Before opening a public issue, remove benchmark text, model output, API traces, and private paths. A safe report should contain the affected version, component, exception type, sanitized stack frames, and a minimal structural reproduction.

For a vulnerability that could expose credentials or private outputs, contact the repository owner privately instead of filing a public issue.

