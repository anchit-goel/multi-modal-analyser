import warnings
warnings.filterwarnings('ignore')
import os, glob, json
from pathlib import Path
from inference import predict

BASE_DIR = Path(__file__).resolve().parent
files = glob.glob(str(BASE_DIR / 'wav samples' / '*.wav'))
results = {}
for f in files:
    try:
        res = predict(f)
        results[f] = res
    except Exception as e:
        results[f] = {"error": str(e)}

with open(BASE_DIR / 'results.json', 'w') as out:
    json.dump(results, out, indent=2)
