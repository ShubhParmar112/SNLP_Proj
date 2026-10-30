"""Generate the SMOKE-TEST fixture corpus.

    *** THIS IS NOT THE PROJECT DATASET. ***

These comments were written to exercise the pipeline end to end before the real corpus
exists. Any metric computed on them is a smoke test proving the code runs, never a
result. Every table produced from this file is captioned
"SMOKE TEST - NOT A RESULT" by run_baseline.py.

Replace data/raw/annotations.csv with real annotations before reporting anything.

Output: long format, one row per (feedback_id, annotator_id, aspect).
Comments with no aspect get one row with an empty aspect_category.
"""

import csv

from config import FIXTURES

# (text, doc_sentiment, [(aspect, polarity, aspect_term), ...])
FIXTURE = [
    ("The lectures were informative but the pace was too fast.", "negative",
     [("teaching", "positive", "lectures were informative"),
      ("pace", "negative", "the pace was too fast")]),
    ("Assignments were difficult but genuinely useful.", "neutral",
     [("assessment", "conflict", "difficult but genuinely useful")]),
    ("We barely get enough time to finish everything.", "negative",
     [("workload", "negative", "barely get enough time to finish everything")]),
    ("There were four assignments this semester.", "neutral",
     [("assessment", "neutral", "four assignments this semester")]),
    ("Another three-hour lecture, exactly what we needed.", "negative",
     [("pace", "negative", "Another three-hour lecture")]),
    ("Good course.", "positive", []),
    ("Sir is always available for doubts but the slides are unreadable.", "neutral",
     [("faculty_interaction", "positive", "always available for doubts"),
      ("resources", "negative", "the slides are unreadable")]),
    ("The content is relevant, though it is covered too quickly.", "neutral",
     [("course_content", "positive", "content is relevant"),
      ("pace", "negative", "covered too quickly")]),
    ("Please upload the recordings earlier.", "negative",
     [("resources", "negative", "upload the recordings earlier")]),
    ("Not very informative.", "negative",
     [("teaching", "negative", "Not very informative")]),
    ("Concepts were explained clearly with good real world examples.", "positive",
     [("teaching", "positive", "explained clearly with good real world examples")]),
    ("The professor never responds to emails.", "negative",
     [("faculty_interaction", "negative", "never responds to emails")]),
    ("Three submissions in the same week is unreasonable.", "negative",
     [("workload", "negative", "Three submissions in the same week")]),
    ("The syllabus is outdated and not relevant to industry.", "negative",
     [("course_content", "negative", "syllabus is outdated and not relevant")]),
    ("Lab equipment was frequently not working.", "negative",
     [("resources", "negative", "Lab equipment was frequently not working")]),
    ("Exam questions were fair and covered what was taught.", "positive",
     [("assessment", "positive", "Exam questions were fair")]),
    ("Loved the course, learnt a lot.", "positive", []),
    ("The pace was comfortable and easy to follow.", "positive",
     [("pace", "positive", "pace was comfortable and easy to follow")]),
    ("Too many topics squeezed into too few lectures.", "negative",
     [("pace", "negative", "Too many topics squeezed into too few lectures"),
      ("course_content", "negative", "Too many topics")]),
    ("Faculty was approachable and clarified doubts patiently.", "positive",
     [("faculty_interaction", "positive", "approachable and clarified doubts patiently")]),
    ("Grading was inconsistent across sections.", "negative",
     [("assessment", "negative", "Grading was inconsistent")]),
    ("The notes uploaded on the portal were detailed and helpful.", "positive",
     [("resources", "positive", "notes uploaded on the portal were detailed")]),
    ("Content was good but delivery was monotonous.", "neutral",
     [("course_content", "positive", "Content was good"),
      ("teaching", "negative", "delivery was monotonous")]),
    ("Deadlines clashed with other courses.", "negative",
     [("workload", "negative", "Deadlines clashed with other courses")]),
    ("The project was interesting though very time consuming.", "neutral",
     [("assessment", "positive", "project was interesting"),
      ("workload", "negative", "very time consuming")]),
    ("Nothing much to say.", "neutral", []),
    ("Quiz timings were announced at the last moment.", "negative",
     [("assessment", "negative", "Quiz timings were announced at the last moment")]),
    ("Explanations were clear but there were too few examples.", "neutral",
     [("teaching", "conflict", "clear but there were too few examples")]),
    ("The classroom projector rarely worked.", "negative",
     [("resources", "negative", "classroom projector rarely worked")]),
    ("Course structure was logical and well sequenced.", "positive",
     [("course_content", "positive", "structure was logical and well sequenced")]),
    ("Workload was manageable this semester.", "positive",
     [("workload", "positive", "Workload was manageable")]),
    ("We were never told what the assignment rubric was.", "negative",
     [("assessment", "negative", "never told what the assignment rubric was")]),
    ("Teaching was excellent, easily the best course this term.", "positive",
     [("teaching", "positive", "Teaching was excellent")]),
    ("The textbook recommended was hard to obtain.", "negative",
     [("resources", "negative", "textbook recommended was hard to obtain")]),
    ("Sessions moved at a reasonable speed.", "positive",
     [("pace", "positive", "moved at a reasonable speed")]),
    ("Doubt sessions were scheduled but nobody attended them.", "neutral",
     [("faculty_interaction", "neutral", "Doubt sessions were scheduled")]),
    ("The course covers too little depth for a final year elective.", "negative",
     [("course_content", "negative", "too little depth")]),
    ("Assignments helped me understand the concepts better.", "positive",
     [("assessment", "positive", "Assignments helped me understand")]),
    ("Lectures are not engaging at all.", "negative",
     [("teaching", "negative", "not engaging at all")]),
    ("Everything was fine.", "neutral", []),
    ("Slides are uploaded on time every week.", "positive",
     [("resources", "positive", "uploaded on time every week")]),
    ("The instructor rushed through the last three units.", "negative",
     [("pace", "negative", "rushed through the last three units")]),
    ("Exams were much harder than the practice problems.", "negative",
     [("assessment", "negative", "Exams were much harder")]),
    ("Good teaching, poor scheduling.", "neutral",
     [("teaching", "positive", "Good teaching"),
      ("workload", "negative", "poor scheduling")]),
    ("The lab component was the most useful part of the course.", "positive",
     [("resources", "positive", "lab component was the most useful part")]),
    ("Too much theory and not enough practice.", "negative",
     [("course_content", "negative", "Too much theory and not enough practice")]),
    ("Faculty explained patiently even when asked repeatedly.", "positive",
     [("faculty_interaction", "positive", "explained patiently even when asked repeatedly")]),
    ("It was okay.", "neutral", []),
    ("Assignment deadlines were extended when we requested.", "positive",
     [("faculty_interaction", "positive", "deadlines were extended when we requested"),
      ("workload", "positive", "deadlines were extended")]),
    ("Concepts are good but the pace and assignments are difficult.", "negative",
     [("course_content", "positive", "Concepts are good"),
      ("pace", "negative", "the pace"),
      ("assessment", "negative", "assignments are difficult")]),
    ("No recordings were shared for missed classes.", "negative",
     [("resources", "negative", "No recordings were shared")]),
    ("The subject is interesting but poorly organised.", "neutral",
     [("course_content", "conflict", "interesting but poorly organised")]),
    ("More practice problems would help a lot.", "negative",
     [("resources", "negative", "More practice problems would help")]),
    ("Attendance policy was strict but fair.", "neutral",
     [("other", "conflict", "Attendance policy was strict but fair")]),
    ("The viva was conducted professionally.", "positive",
     [("assessment", "positive", "viva was conducted professionally")]),
    ("Could not keep up with the weekly submissions.", "negative",
     [("workload", "negative", "Could not keep up with the weekly submissions")]),
    ("Lectures were clear, assignments were clear, exams were clear.", "positive",
     [("teaching", "positive", "Lectures were clear"),
      ("assessment", "positive", "assignments were clear")]),
    ("The course needs more industry case studies.", "negative",
     [("course_content", "negative", "needs more industry case studies")]),
    ("Sir explains well but does not reply to messages.", "neutral",
     [("teaching", "positive", "explains well"),
      ("faculty_interaction", "negative", "does not reply to messages")]),
    ("Best organised course I have taken.", "positive",
     [("course_content", "positive", "Best organised course")]),
]


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (text, doc_sent, aspects) in enumerate(FIXTURE, start=1):
        fid = f"FIX{i:03d}"
        if not aspects:
            rows.append(dict(feedback_id=fid, feedback_text=text,
                             annotator_id="fixture", doc_sentiment=doc_sent,
                             aspect_category="", polarity="", aspect_term="",
                             span_start="", span_end="", uncertain=0,
                             double_annotated=0, notes=""))
            continue
        for aspect, polarity, term in aspects:
            start = text.find(term)
            end = start + len(term) if start >= 0 else -1
            rows.append(dict(feedback_id=fid, feedback_text=text,
                             annotator_id="fixture", doc_sentiment=doc_sent,
                             aspect_category=aspect, polarity=polarity,
                             aspect_term=term if start >= 0 else "",
                             span_start=start if start >= 0 else -1,
                             span_end=end if start >= 0 else -1,
                             uncertain=0, double_annotated=0, notes=""))

    out = FIXTURES / "smoke_test_annotations.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_comments = len({r["feedback_id"] for r in rows})
    print(f"Wrote {out}")
    print(f"  {n_comments} comments, {len(rows)} annotation rows")
    print("  *** SMOKE TEST FIXTURE - NOT THE PROJECT DATASET ***")




def make_calibration_set(n=40, disagree_rate=0.20, seed=42):
    """Double-annotated calibration fixture so src/iaa.py can be tested before real
    annotation begins. Annotator B's labels are perturbed to create realistic
    disagreement. NOT real annotation data."""
    import random
    rng = random.Random(seed)
    from config import ASPECTS, POLARITIES

    rows = []
    for i, (text, doc_sent, aspects) in enumerate(FIXTURE[:n], start=1):
        fid = f"CAL{i:03d}"
        for ann in ("A_E029", "B_E030"):
            items = list(aspects)
            if ann.startswith("B") and items and rng.random() < disagree_rate:
                j = rng.randrange(len(items))
                a, p, t = items[j]
                if rng.random() < 0.5:
                    items[j] = (rng.choice([x for x in ASPECTS if x != a]), p, t)
                else:
                    items[j] = (a, rng.choice([x for x in POLARITIES if x != p]), t)
            if not items:
                rows.append(dict(feedback_id=fid, feedback_text=text, annotator_id=ann,
                                 doc_sentiment=doc_sent, aspect_category="", polarity="",
                                 aspect_term="", span_start="", span_end="", uncertain=0,
                                 double_annotated=1, notes=""))
                continue
            for aspect, polarity, term in items:
                start = text.find(term)
                end = start + len(term) if start >= 0 else -1
                rows.append(dict(feedback_id=fid, feedback_text=text, annotator_id=ann,
                                 doc_sentiment=doc_sent, aspect_category=aspect,
                                 polarity=polarity,
                                 aspect_term=term if start >= 0 else "",
                                 span_start=start if start >= 0 else -1,
                                 span_end=end if start >= 0 else -1,
                                 uncertain=0, double_annotated=1, notes=""))

    out = FIXTURES / "calibration_set_annotated.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}  ({len(rows)} rows, 2 annotators)")


if __name__ == "__main__":
    main()
    make_calibration_set()
