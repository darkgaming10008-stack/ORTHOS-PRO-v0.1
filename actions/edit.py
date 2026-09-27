"""
Edit tool - Modify files using precise string replacement.
Supports single replacement and replace-all functionality.
"""
from pathlib import Path
from typing import Optional


def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> dict:
    """
    Edit a file by replacing exact text with new text.

    Args:
        file_path: Full path to the file to edit
        old_string: Exact text to find and replace
        new_string: Text to replace it with
        replace_all: If True, replace all occurrences. If False, replace only first

    Returns:
        Dictionary with operation result:
        {
            "success": bool,
            "file_path": str,
            "replacements": int,       # Number of replacements made
            "message": str,            # Human-readable message
            "error": str or None
        }
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "file_path": file_path,
                "replacements": 0,
                "message": "",
                "error": f"File not found: {file_path}"
            }

        if not path.is_file():
            return {
                "success": False,
                "file_path": file_path,
                "replacements": 0,
                "message": "",
                "error": f"Path is not a file: {file_path}"
            }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        if old_string not in content:
            return {
                "success": False,
                "file_path": file_path,
                "replacements": 0,
                "message": "",
                "error": f"Text not found in file. Make sure the old_string matches exactly, including whitespace and newlines."
            }

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = content.count(old_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return {
            "success": True,
            "file_path": str(path.resolve()),
            "replacements": replacements,
            "message": f"Successfully made {replacements} replacement(s) in {file_path}",
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "file_path": file_path,
            "replacements": 0,
            "message": "",
            "error": str(e)
        }


def edit_tool(parameters: dict, response=None, player=None, **kwargs) -> str:
    """
    Tool-compatible wrapper for edit function.
    Returns JSON string result.
    """
    import json

    file_path = parameters.get("file_path")
    old_string = parameters.get("old_string")
    new_string = parameters.get("new_string")

    if not file_path:
        return json.dumps({
            "success": False,
            "replacements": 0,
            "error": "file_path is required"
        })

    if old_string is None:
        return json.dumps({
            "success": False,
            "replacements": 0,
            "error": "old_string is required"
        })

    if new_string is None:
        return json.dumps({
            "success": False,
            "replacements": 0,
            "error": "new_string is required"
        })

    replace_all = parameters.get("replace_all", False)

    result = edit(file_path, old_string, new_string, replace_all)
    return json.dumps(result, indent=2)
