"""Week 8: single-comment latency of the asset on this machine's CPU.

    python scripts/measure_cpu_latency.py        # writes results/week8_cpu_latency.json

In-process (no HTTP), the same call the API makes. 5 warm-up calls, then 50 timed calls
on the handout example sentence.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS
from pipeline import FeedbackAnalyzer

TEXT = ("The lectures were very informative, but the pace was too fast and the assignments "
        "were difficult.")

fa = FeedbackAnalyzer(device="cpu")
for _ in range(5):
    fa.analyze(TEXT)
times = []
for _ in range(50):
    t0 = time.perf_counter()
    fa.analyze(TEXT)
    times.append(1000 * (time.perf_counter() - t0))

out = {"device": "cpu", "processor": platform.processor(), "calls": len(times),
       "mean_ms": round(statistics.mean(times), 1), "median_ms": round(statistics.median(times), 1),
       "min_ms": round(min(times), 1), "max_ms": round(max(times), 1),
       "model_version": fa.model_version}
(RESULTS / "week8_cpu_latency.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
