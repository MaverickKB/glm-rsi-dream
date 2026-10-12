"""The packaged identity and immutable runtime contract."""
import json
from pathlib import Path

MANIFEST = json.loads((Path(__file__).with_name('resources') / 'manifest.json').read_text())
