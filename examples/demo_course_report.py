"""Demo: a course report from 150 real student comments, printed for people to read.

    python src/api.py                       # first, in another terminal
    python examples/demo_course_report.py

Sends examples/demo_comments.json (150 public reviews from the test split) to
POST /v1/analyze-feedback/batch and prints what a course coordinator would see.
Standard library only.
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
batch = json.loads((Path(__file__).parent / "demo_comments.json").read_text(encoding="utf-8"))
body = {"course_id": batch["course_id"], "items": batch["items"],
        "options": {"include_examples": True, "max_examples_per_aspect": 2}}
req = urllib.request.Request(BASE + "/v1/analyze-feedback/batch", method="POST",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
rep = json.loads(urllib.request.urlopen(req, timeout=120).read())

print(f"COURSE REPORT  {rep['course_id']}   ({rep['n_analyzed']} comments, "
      f"model {rep['model_version']})\n")
sd = rep["sentiment_distribution"]
print(f"Overall mood: {sd['positive']} positive, {sd['neutral']} neutral, "
      f"{sd['negative']} negative\n")
by = {s["aspect"]: s for s in rep["aspect_summary"]}


def show(title, names):
    print(title)
    if not names:
        print("   (none)")
    for a in names:
        s = by[a]
        print(f"   {a:<20} {s['mentions']:>3} mentions   net {s['net_sentiment']:+.2f}")
        for q in s["representative_examples"]:
            print(f"        \"{q}\"")
    print()


show("TOP STRENGTHS", rep["top_positive_aspects"])
show("TOP CONCERNS", rep["top_concerns"])
show(f"EMERGING CONCERNS (fewer than {rep['min_mentions_for_ranking']} mentions, but negative)",
     rep["emerging_concerns"])
print("Every aspect:")
for s in rep["aspect_summary"]:
    d = s["polarity_distribution"]
    print(f"   {s['aspect']:<20} {s['mentions']:>3}   +{d['positive']} -{d['negative']} "
          f"neutral {d['neutral']} mixed {d['conflict']}")
print("\nCaveats shown with every report:")
for c in rep["caveats"]:
    print("   - " + c)
