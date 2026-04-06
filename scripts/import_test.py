import sys, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "modules" / "multimodal_injection"))
print('STARTING IMPORT')
try:
    import multimodal_detector
    print('MODULE IMPORTED')
    print('HAS_ANALYZE', hasattr(multimodal_detector, 'analyze_multimodal'))
except Exception:
    traceback.print_exc()
