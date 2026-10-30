"""Demo: what Group 6 (Early Warning) receives from us - course-level signals only.

    python src/api.py                       # first, in another terminal
    python examples/demo_support_signals.py

Sends the same 150 demo comments as demo_course_report.py to POST /v1/support-signals
and prints the signals as Group 6 would receive them. Standard library only.
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
batch = json.loads((Path(__file__).parent / "demo_comments.json").read_text(encoding="utf-8"))
body = {"course_id": batch["course_id"], "items": batch["items"]}
req = urllib.request.Request(BASE + "/v1/support-signals", method="POST",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(req, timeout=120).read())

print(f"SUPPORT SIGNALS FOR GROUP 6  {out['course_id']}   ({out['n_analyzed']} comments, "
      f"model {out['model_version']})\n")
if not out["signals"]:
    print("   (no signal: no aspect is a top or emerging concern)")
for s in out["signals"]:
    print(f"   {s['aspect']:<20} {s['basis']:<17} {s['mention_count']:>3} negative of "
          f"{s['total_comments']} comments ({s['proportion']:.1%})")
print("\nFirst signal exactly as sent:")
print(json.dumps(out["signals"][0] if out["signals"] else {}, indent=2))
print("\nCaveats sent with every response:")
for c in out["caveats"]:
    print("   - " + c)
