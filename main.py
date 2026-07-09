"""Compatibility entry point for EdgeLLM-Lab.

Prefer using:

    python3 -m cli smoke
    python3 -m cli train -c configs/tiny_gpt.yaml

or, after installing the project:

    edgellm smoke
"""

from __future__ import annotations

from cli import main


if __name__ == "__main__":
    raise SystemExit(main())
