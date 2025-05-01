import json
from typing import List, Dict
 
def load_entries(filepath: str) -> List[Dict]:
    with open(filepath, "r") as f:
        return json.load(f) 