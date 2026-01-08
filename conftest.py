import sys
from pathlib import Path

# Ensures repo root is on sys.path so pytest can import `research`
# This repo is a research project, not an installed package.

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
