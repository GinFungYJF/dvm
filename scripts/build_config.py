"""Update the Make configuration stamp only when settings change."""
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
content = json.dumps(sys.argv[2:]) + "\n"
if not path.exists() or path.read_text() != content:
    path.write_text(content)
