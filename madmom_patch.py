#!/usr/bin/env python3
"""
madmom_patch.py
----------------
Fixes Python 3.10+ compatibility issue in madmom by replacing:
    from collections import MutableSequence
with:
    from collections.abc import MutableSequence

Run this script **once after installing dependencies**:
    python madmom_patch.py
"""

import sys
import os
import importlib.util

print("🔍 Locating madmom installation...")

# Find the madmom module path without importing it
spec = importlib.util.find_spec("madmom")
if spec is None or not spec.origin:
    print("⚠️ madmom is not installed. Please run:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

madmom_dir = os.path.dirname(spec.origin)
processors_file = os.path.join(madmom_dir, "processors.py")

if not os.path.isfile(processors_file):
    print(f"❌ processors.py not found at: {processors_file}")
    sys.exit(1)

try:
    with open(processors_file, "r", encoding="utf-8") as f:
        content = f.read()

    if "from collections import MutableSequence" not in content:
        print("ℹ️ Patch not needed. madmom is already compatible.")
        sys.exit(0)

    content = content.replace(
        "from collections import MutableSequence",
        "from collections.abc import MutableSequence"
    )

    with open(processors_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Successfully patched madmom at:\n   {processors_file}")

except Exception as e:
    print(f"❌ Failed to patch madmom: {e}")
    sys.exit(1)
