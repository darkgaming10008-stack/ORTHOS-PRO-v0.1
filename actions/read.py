"""
Read tool - Read file contents with optional line range support.
"""
from pathlib import Path
from typing import Optional


def read(
    file_path: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Read file contents from the local filesystem.

    Args:
        file_path: Full path to the file to read
        offset: Line number to start reading from (1-indexed). If None, starts from line 1
        limit: Maximum number of lines to read. If None, reads until end of file

    Returns:
        Dictionary with file content and metadata:
        {
            "success": bool,
            "content": str,           # File contents
            "file_path": str,         # Path that was read
            "total_lines": int,       # Total lines in file
            "lines_read": int,        # Number of lines returned
            "error": str or None      # Error message if failed
        }
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "content": "",
                "file_path": file_path,
                "total_lines": 0,
                "lines_read": 0,
                "error": f"File not found: {file_path}"
            }

        if not path.is_file():
            return {
                "success": False,
                "content": "",
                "file_path": file_path,
                "total_lines": 0,
                "lines_read": 0,
                "error": f"Path is not a file: {file_path}"
            }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        total_lines = len(lines)

        start_idx = 0
        if offset is not None:
            start_idx = max(0, offset - 1)

        end_idx = total_lines
        if limit is not None:
            end_idx = min(start_idx + limit, total_lines)

        selected_lines = lines[start_idx:end_idx]
        content = ''.join(selected_lines)

        lines_read = len(selected_lines)

        return {
            "success": True,
            "content": content,
            "file_path": str(path.resolve()),
            "total_lines": total_lines,
            "lines_read": lines_read,
            "offset": start_idx + 1,
            "limit": limit,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "content": "",
            "file_path": file_path,
            "total_lines": 0,
            "lines_read": 0,
            "error": str(e)
        }


def read_tool(parameters: dict, response=None, player=None, **kwargs) -> str:
    """
    Tool-compatible wrapper for read function.
    Returns JSON string result.
    """
    import json

    file_path = parameters.get("file_path")
    if not file_path:
        return json.dumps({
            "success": False,
            "error": "file_path is required"
        })

    offset = parameters.get("offset")
    limit = parameters.get("limit")

    result = read(file_path, offset, limit)
    return json.dumps(result, indent=2)
