"""Final architecture diagram (handout §15 H): input -> preprocessing -> NLP model ->
post-processing -> API / asset -> applications.

    python scripts/draw_architecture.py          # writes docs/architecture.png

Drawn from code so it can be regenerated whenever the system changes. The Week 1-2
design diagram it replaces is kept in the earlier snapshot folders.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.png"
NAVY, BLUE, GREY, INK = "#1F3864", "#2E5C8A", "#595959", "#222222"
FILL = {"stage": "#DEE6F0", "model": "#C9D7EA", "app": "#EEF1F5", "g6": "#F4E9D8"}

# (stage label, box title, detail lines, fill)
LAYERS = [
    ("INPUT", "Student feedback",
     ["one anonymised comment, or up to 500 comments for one course (JSON)"], "stage"),
    ("PREPROCESSING", "Validation  ->  PII redaction",
     ["empty / over 5,000 characters rejected with an error code",
      "emails, phone numbers, roll numbers, titled names replaced; casing and punctuation kept"],
     "stage"),
    ("NLP MODEL", "DeBERTa-v3-base with aspect attention",
     ["encoder -> 8 learned aspect queries, each attending to its own words",
      "per aspect: none / positive / negative / neutral / conflict;  [CLS] -> document sentiment",
      "trained on EduRABSA (4,198 comments) with evidence-span supervision"], "model"),
    ("POST-PROCESSING", "Decisions, calibration, evidence, aggregation",
     ["per-aspect presence thresholds (tuned on validation, frozen)",
      "isotonic calibration -> calibrated_probability;  low_confidence warnings",
      "evidence span grown to its clause;  course report: ranking floor, emerging concerns, caveats"],
     "stage"),
    ("API / ASSET", "FastAPI service  (src/api.py)",
     ["POST /v1/analyze-feedback    POST /v1/analyze-feedback/batch    POST /v1/support-signals",
      "GET /v1/health    GET /v1/schema    GET /v1/model-card    interactive docs at /docs"],
     "model"),
]
APPS = [
    ("Course coordinator", "course report: strengths,\nconcerns, quotes, caveats", "app"),
    ("Group 6: Early Warning", "course-level concern signals +\ncomment handoff (advisory only)", "g6"),
    ("Any other application", "HTTP + JSON, standard library\nclient (examples/)", "app"),
]

W, H = 12.0, 13.2
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, fill, title, lines, title_size=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[fill], ec=BLUE, lw=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=NAVY)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - 0.34 * i, line, ha="center", va="top",
                fontsize=9.6, color=INK)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=BLUE))


ax.text(W / 2, H - 0.25, "Student Feedback Intelligence: system architecture", ha="center",
        va="top", fontsize=17, fontweight="bold", color=NAVY)
ax.text(W / 2, H - 0.8, "Group 5, SNLP Course Project 2026  ·  model transformer-absa-0.2.2  ·  "
        "dataset v1.1-edurabsa", ha="center", va="top", fontsize=10, color=GREY)

x0, bw, gap = 2.3, 9.4, 0.38
y = H - 1.35
for stage, title, lines, fill in LAYERS:
    h = 0.95 + 0.34 * len(lines)
    y -= h
    ax.text(x0 - 0.2, y + h / 2, stage, ha="right", va="center", fontsize=10,
            fontweight="bold", color=GREY)
    box(x0, y, bw, h, fill, title, lines)
    api_bottom = y
    if stage != LAYERS[-1][0]:
        y -= gap
        arrow(x0 + bw / 2, api_bottom - 0.02, x0 + bw / 2, y + 0.02)

y = api_bottom - 0.7                       # top of the applications row
ax.text(x0 - 0.2, y - 0.75, "APPLICATIONS", ha="right", va="center", fontsize=10,
        fontweight="bold", color=GREY)
aw, ag = (bw - 2 * 0.3) / 3, 0.3
for k, (title, detail, fill) in enumerate(APPS):
    ax_x = x0 + k * (aw + ag)
    box(ax_x, y - 1.5, aw, 1.5, fill, title, detail.split("\n"), title_size=11)
    arrow(x0 + bw / 2, api_bottom - 0.02, ax_x + aw / 2, y + 0.02)

ax.text(x0 + bw / 2, y - 1.95, "Outputs are advisory and course-level: never a basis for "
        "decisions about an individual student or staff member.", ha="center", va="top",
        fontsize=10, style="italic", color=GREY)
ax.set_ylim(y - 2.4, H)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
