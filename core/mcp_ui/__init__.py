"""MCP UI — Modern, card-based MCP management interface for Orthos.

Architecture:
- styles.py        Centralized theme, colors, animations
- components.py    Reusable widgets: cards, toasts, spinners, badges, shimmer
- server_panel.py  Card-based server grid with health monitoring
- install_flow.py  Enhanced install dialog with test connection, docs, version selector
- search.py        AI-powered natural language search + smart recommendations
- health.py        Server health monitoring, uptime, response time
- sidebar.py       Right sidebar panel integration for main Orthos UI
- voice.py         Voice command handlers for MCP management
- groups.py        Server groups, favorites, bulk operations, import/export
- security.py      Keychain integration, permission prompts, audit log
- __init__.py      Public API, legacy compatibility
"""
