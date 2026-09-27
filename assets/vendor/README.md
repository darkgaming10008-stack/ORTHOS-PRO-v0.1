# Orthos bundled chat-renderer assets

The activity log must work without network access, so these browser assets are
checked into the application instead of being loaded from a CDN.

- KaTeX 0.16.25 (`katex/`), MIT licensed
- Marked 15.0.12 (`marked/`), MIT licensed

Only their production browser distributions are used by `ui.py`.  KaTeX fonts
are included because its stylesheet references them at runtime.
