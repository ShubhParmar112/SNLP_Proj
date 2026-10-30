"""How another group calls the Student Feedback Intelligence API.

Start the service first (on the machine that has the model):
    python src/api.py

Then, from anywhere that can reach it:
    python examples/client_example.py [http://127.0.0.1:8000]

Standard library only - no packages to install on the calling side.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:            # 4xx / 5xx carry the error JSON
        return e.code, json.loads(e.read())


# 1. Is the service up?
print("health:", call("GET", "/v1/health")[1])

# 2. One comment -> aspects, polarity, calibrated scores, evidence
status, out = call("POST", "/v1/analyze-feedback", {
    "feedback_id": "F001",
    "text": "The lectures were very informative, but the pace was too fast "
            "and the assignments were difficult."})
print("\nanalyze-feedback", status)
print(json.dumps(out, indent=2))

# 3. A batch of comments -> course report
comments = [
    "Great teacher, explains every concept with real examples.",
    "The pace was far too fast, I could not keep up.",
    "Too many assignments every single week.",
    "The weekly assignments took most of my weekend.",
    "Slides are clear and the notes are really useful.",
    "Workload is heavy but the professor is always willing to help.",
]
status, rep = call("POST", "/v1/analyze-feedback/batch", {
    "course_id": "DEMO-101",
    "items": [{"feedback_id": f"C{i + 1}", "text": t} for i, t in enumerate(comments)]})
print("\nanalyze-feedback/batch", status)
print(json.dumps({k: rep[k] for k in ("n_analyzed", "top_positive_aspects", "top_concerns",
                                      "emerging_concerns", "caveats")}, indent=2))

# 4. Errors come back in one documented format
print("\nempty text ->", call("POST", "/v1/analyze-feedback", {"text": "  "}))

# 5. Speed of a single round trip, as another application would see it
t0 = time.time()
for _ in range(20):
    call("POST", "/v1/analyze-feedback", {"text": "Lectures were good but exams were hard."})
print(f"\nmean round trip over 20 calls: {1000 * (time.time() - t0) / 20:.0f} ms")
