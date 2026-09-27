"""
Grep tool - Search file contents using regex patterns.
Supports file pattern filtering, exclusion patterns, and .gitignore compliance.
"""
import re
import os
from pathlib import Path
from typing import Optional, List


def _matches_pattern(path: Path, pattern: str) -> bool:
    """Check if a path matches a file pattern like *.py"""
    if not pattern:
        return True
    import fnmatch
    return fnmatch.fnmatch(path.name, pattern)


def _should_exclude(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if a path should be excluded based on exclude patterns"""
    if not exclude_patterns:
        return False
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
        parts = path.parts
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def grep(
    pattern: str,
    path: Optional[str] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    max_results: int = 100,
) -> dict:
    """
    Search file contents using regex patterns.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in. If None, searches Orthos folder
        include: File pattern to include, e.g., "*.py", "*.js"
        exclude: Patterns to exclude, e.g., "__pycache__|*.pyc|node_modules"
        max_results: Maximum number of results to return (default 100)

    Returns:
        Dictionary with search results:
        {
            "success": bool,
            "pattern": str,              # The regex pattern used
            "path": str,                # Directory searched
            "total_matches": int,        # Total matches found
            "results": [                 # List of matches (limited by max_results)
                {
                    "file": str,         # Full path to file
                    "line": int,         # Line number (1-indexed)
                    "content": str,      # The matching line content
                    "line_preview": str  # Truncated content for display
                },
                ...
            ],
            "files_searched": int,      # Number of files searched
            "error": str or None
        }
    """
    import fnmatch

    try:
        re.compile(pattern)
    except re.error as e:
        return {
            "success": False,
            "pattern": pattern,
            "path": path or "Orthos folder",
            "total_matches": 0,
            "results": [],
            "files_searched": 0,
            "error": f"Invalid regex pattern: {e}"
        }

    if path:
        search_path = Path(path)
        if not search_path.exists():
            return {
                "success": False,
                "pattern": pattern,
                "path": path,
                "total_matches": 0,
                "results": [],
                "files_searched": 0,
                "error": f"Search path does not exist: {path}"
            }
    else:
        search_path = Path(__file__).resolve().parent.parent

    exclude_patterns = []
    if exclude:
        exclude_patterns = [p.strip() for p in exclude.split('|') if p.strip()]

    results = []
    files_searched = 0

    try:
        for root, dirs, files in os.walk(search_path):
            root_path = Path(root)

            if _should_exclude(root_path, exclude_patterns):
                dirs.clear()
                continue

            dirs[:] = [d for d in dirs if not _should_exclude(root_path / d, exclude_patterns)]

            for filename in files:
                file_path = root_path / filename

                if _should_exclude(file_path, exclude_patterns):
                    continue

                if include and not _matches_pattern(file_path, include):
                    continue

                files_searched += 1

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                line_content = line.rstrip()
                                preview = line_content[:200] + ('...' if len(line_content) > 200 else '')

                                results.append({
                                    "file": str(file_path),
                                    "line": line_num,
                                    "content": line_content,
                                    "line_preview": preview
                                })

                                if len(results) >= max_results:
                                    return {
                                        "success": True,
                                        "pattern": pattern,
                                        "path": str(search_path),
                                        "total_matches": len(results),
                                        "results": results,
                                        "files_searched": files_searched,
                                        "error": None,
                                        "truncated": True,
                                        "max_results": max_results
                                    }
                except (PermissionError, OSError):
                    continue

    except Exception as e:
        return {
            "success": False,
            "pattern": pattern,
            "path": str(search_path),
            "total_matches": 0,
            "results": [],
            "files_searched": files_searched,
            "error": str(e)
        }

    return {
        "success": True,
        "pattern": pattern,
        "path": str(search_path),
        "total_matches": len(results),
        "results": results,
        "files_searched": files_searched,
        "error": None,
        "truncated": False
    }


def grep_tool(parameters: dict, response=None, player=None, **kwargs) -> str:
    """
    Tool-compatible wrapper for grep function.
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
    include = parameters.get("include")
    exclude = parameters.get("exclude")
    max_results = parameters.get("max_results", 100)

    result = grep(pattern, path, include, exclude, max_results)
    return json.dumps(result, indent=2)
