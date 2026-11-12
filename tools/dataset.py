"""Read the exact sensor precision embedded in the generated static page."""
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def load_dataset():
    html = (ROOT / 'index.html').read_text()
    def read(name):
        start = re.search(r'\b' + name + r'\s*=', html).end()
        return json.JSONDecoder().raw_decode(html[start:])[0]
    return read('ALL'), read('META')
