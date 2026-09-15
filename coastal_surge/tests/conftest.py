import sys
from pathlib import Path

# The pipeline scripts import each other with sibling-style `import
# nc_metadata` (they're run as standalone scripts, not a package), so tests
# need coastal_surge/ itself on sys.path, not just the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
