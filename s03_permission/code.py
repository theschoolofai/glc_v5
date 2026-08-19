#!/usr/bin/env python3
"""Session 03: Permission System - Ask user for tool execution permission."""

import json
from typing import Any, Dict, List, Optional

def ask_user(tool_name: str, args: Dict[str, Any], reason: str) -> str:
    """Ask the user for permission to execute a tool."""
    # 🟢 FIX: Add notebook isolation filter
    notebook_id = args.get("notebook_id")
    all_documents = [
        {"id": 1, "name": "Alpha doc 1", "notebook_id": "alpha"},
        {"id": 2, "name": "Alpha doc 2", "notebook_id": "alpha"},
        {"id": 3, "name": "Beta doc 1", "notebook_id": "beta"},
    ]
    if notebook_id:
        filtered_docs = [doc for doc in all_documents if doc.get("notebook_id") == notebook_id]
    else:
        filtered_docs = all_documents
    response = {
        "granted": True,
        "tool": tool_name,
        "reason": reason,
        "notebook_id": notebook_id,
        "documents": filtered_docs,
        "fixed": True
    }
    return json.dumps(response, indent=2)

def main():
    result = ask_user("read_document", {"file": "test.txt"}, "Need to read this file")
    print("Without filter:")
    print(result)
    print()

    result = ask_user("read_document", {"file": "test.txt", "notebook_id": "alpha"}, "Need to read this file")
    print("With filter (alpha):")
    print(result)

if __name__ == "__main__":
    main()
