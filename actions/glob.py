"""
Glob tool - Find files by glob patterns.
Supports recursive patterns like **/*.py and any directory path.
"""
import os
from pathlib import Path
from typing import Optional, List


def glob(
    pattern: str,
    path: Optional[str] = None,
) -> dict:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py", "*.js", "src/**/*.ts")
        path: Directory to search in. If None, searches Orthos folder

    Returns:
        Dictionary with matching files:
        {
            "success": bool,
            "pattern": str,              # The glob pattern used
            "path": str,                # Directory searched
            "total_matches": int,        # Number of matching files
            "files": [                  # List of matching file paths
                {
                    "path": str,         # Full path to file
                    "name": str,         # File name
                    "size": int,         # File size in bytes
                    "modified": str       # Last modified time (ISO format)
                },
                ...
            ],
            "error": str or None
        }
    """
    if path:
        search_path = Path(path)
        if not search_path.exists():
            return {
                "success": False,
                "pattern": pattern,
                "path": path,
                "total_matches": 0,
                "files": [],
                "error": f"Search path does not exist: {path}"
            }
        if not search_path.is_dir():
            return {
                "success": False,
                "pattern": pattern,
                "path": path,
                "total_matches": 0,
                "files": [],
                "error": f"Path is not a directory: {path}"
            }
    else:
        search_path = Path(__file__).resolve().parent.parent

    if not pattern:
        return {
            "success": False,
            "pattern": pattern,
            "path": str(search_path),
            "total_matches": 0,
            "files": [],
            "error": "pattern is required"
        }

    try:
        matching_paths = list(search_path.glob(pattern))

        matching_paths = [p for p in matching_paths if p.is_file()]

        matching_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        files = []
        for p in matching_paths:
            try:
                stat = p.stat()
                files.append({
                    "path": str(p.resolve()),
                    "name": p.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except OSError:
                continue

        return {
            "success": True,
            "pattern": pattern,
            "path": str(search_path),
            "total_matches": len(files),
            "files": files,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "pattern": pattern,
            "path": str(search_path),
            "total_matches": 0,
            "files": [],
            "error": str(e)
        }


def glob_tool(parameters: dict, response=None, player=None, **kwargs) -> str:
    """
    Tool-compatible wrapper for glob function.
    Returns JSON string result.
    """
    import json

    pattern = parameters.get("pattern")
    if not pattern:
        return json.dumps({
            "success": False,
            "error": "pattern is required"
        })

    path = parameters.get("path")

    result = glob(pattern, path)
    return json.dumps(result, indent=2)
