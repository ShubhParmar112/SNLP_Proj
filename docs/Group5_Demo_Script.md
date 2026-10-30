---
title: Student Feedback Intelligence
subtitle: Final Demonstration Script
week: Final deliverable D: demonstration (31 October 2026)
course: SNLP Course Project 2026   •   Group 5
members: E029  ·  E030  ·  E031  ·  E033  ·  E034  ·  E035
version: Document version 1.0  •  Model transformer-absa-0.2.2  •  API v1
note: Every expected output below was recorded on the demo laptop on 30 October (results/week8_demo_*.txt).
footer: Group 5  •  Student Feedback Intelligence  •  Demo Script  •  
---

# Final Demonstration Script

## 1.  Before the session (10 minutes earlier)

Open **two PowerShell windows**. In both, go to the project:

```
cd "C:\SNLP\SNLP Sem 11 Project\Group5_Student_Feedback_Intelligence (till week 7-8)\Group5_Student_Feedback_Intelligence"
.venv\Scripts\activate
```

**Window 1** starts the service: `python src\api.py`. Wait for `Uvicorn running on http://127.0.0.1:8000`; the model loads first. A warning about a “tokenizer … incorrect regex pattern” is harmless.

Open the browser at **`http://127.0.0.1:8000/docs`**. The bare address shows “Not Found”; that is normal. The `/docs` page loads its styling from the internet. If the room has no internet, skip the browser and use the terminal commands in Section 4.

Check: `http://127.0.0.1:8000/v1/health` shows `"status": "ok"`.

## 2.  Running order (fits slide 8, about 3 minutes)

| Step | Show | Say |
|---|---|---|
| 1 | `/docs` page, the list of endpoints | “Any group can call this over HTTP. Every endpoint is documented and can be tried here.” |
| 2 | `POST /v1/analyze-feedback` → Try it out → example **“Handout example (works)”** → Execute | “The handout's own sentence. Three aspects, each with its polarity, the words that show it, and a calibrated confidence.” |
| 3 | **Window 2:** `python examples\demo_course_report.py` | “150 real comments treated as one course: top strengths, the top concern, and an emerging concern that is rare but negative.” |
| 4 | **Window 2:** `python examples\demo_support_signals.py` | “What Group 6 receives: course-level signals only, flagged advisory, with no comment and no id.” |
| 5 | `/docs` → example **“Sarcasm (known failure)”** → Execute | “Our required failure. Sarcasm is read literally, so nothing is found. Sarcasm and Hinglish never appear in the English training reviews.” |

## 3.  What you should see

**Step 2**, the handout example (`results/week8_demo_single_comments.txt`):

```
teaching    positive   aspect 0.9128  polarity 0.5021  "The lectures were very informative"
pace        negative   aspect 0.9559  polarity 0.9804  "the pace was too fast"
assessment  negative   aspect 0.9941  polarity 0.9804  "the assignments were difficult"
```

Point out honestly that the model is sure the *lectures* are discussed (0.91), but only 50/50 on whether that remark is positive, because the sentence is negative overall.

**Step 3**, the course report (`results/week8_demo_course_report.txt`):

- 150 comments: 77 positive, 29 neutral, 44 negative.
- Top strengths: teaching (79 mentions), course_content (64), faculty_interaction (44).
- Top concern: workload, 14 mentions, net −0.71.
- Emerging concern: pace, 3 mentions, all negative.

**Step 4**, Group 6 signals (`results/week7_support_signals_demo.txt`): workload as a top concern (12 negative of 150, 8.0%), pace as an emerging concern (3 of 150), each with `advisory_only: true` and `not_for_individual_action: true`.

**Step 5**, the failure: “Oh great, another surprise quiz on Monday.” returns **no aspects**. The correct answer is assessment: negative. To show a second failure, use **“Hinglish (known failure)”**: “Sir ka padhane ka tareeka bahut accha hai.” also returns no aspects. Its document sentiment comes out positive, so the model senses the tone but cannot place it.

## 4.  If something goes wrong

| Problem | Do this |
|---|---|
| `/docs` looks unstyled or blank (no internet) | Window 2: `python src\pipeline.py "The lectures were very informative, but the pace was too fast and the assignments were difficult."` prints the same JSON |
| Port 8000 already in use | Close any other window still running the API, then start it again |
| `/v1/health` says the model is unavailable | The model folder is missing: `models\deberta-v3-base_sw0.5_s42_d1.1\model.pt` must exist |
| Slow responses | Normal on this laptop's CPU, which has no GPU: about 0.2 s per comment (`results/week8_cpu_latency.json`) |

## 5.  Questions to expect

- **“Why not just ask ChatGPT?”** The handout rules out a prompt-to-LLM wrapper. Ours is a trained, measured model. It runs locally, so no student text leaves the machine, and every judgement points to words in the comment.
- **“How do you know it works?”** It was tested on 900 comments it never saw, scoring 0.704 tuple-F1 against 0.452 for the baseline, with confidence intervals and a frozen test split.
- **“Where does it fail?”** Neutral and mixed opinions, sarcasm, Hinglish, and a bias that blames the teacher (slide 10).
- **“Group 6?”** We serve them course-level signals, and we sent them 38 comments in their own format. Report honestly whether their reply arrived.
