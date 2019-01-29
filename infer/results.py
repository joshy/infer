from pathlib import Path
import json

def list():
    files = [x for x in Path("results").glob("*.json") if x.is_file()]
    result = []
    for f in files:
        with open(f, 'r') as f:
            result.append(json.load(f))

    return result