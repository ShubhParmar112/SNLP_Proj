// Group 5 final presentation (handout §16: 10 slides, 10 minutes).
// Every number is copied from the final report, which traces each one to results/.
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const PROJECT = process.env.PROJECT_DIR;
const OUT = process.argv[2];

const THEME = {
  name: "Group5 Feedback",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A2F", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "EEF4F3",
    accent1: "0F3D3E", accent2: "2A9D8F", accent3: "D1603D", accent4: "E9A23B",
    accent5: "5C6F73", accent6: "B8D8D3", hlink: "2A9D8F", folHlink: "5C6F73",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Student Feedback Intelligence";
pres.author = "Group 5";
const C = pres.SchemeColor;

// --- layouts ----------------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.3,
        fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1,
        valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.0, w: 8.8, h: 1.9,
        fontFace: THEME.bodyFontFace, fontSize: 16, color: C.accent6, valign: "top",
        margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75,
        fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: C.text2,
        valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Group 5  ·  Student Feedback Intelligence", options: {
        x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

pres.defineSlideMaster({
  title: "DARK_TOP",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 8.8, h: 0.9,
        fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.background1,
        valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

let n = 0;
function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  n += 1;
  return s;
}
function tb(slide, text, o) {
  slide.addText(text, { isTextBox: true, margin: 0, fontFace: THEME.bodyFontFace,
    color: C.text1, fontSize: 14, valign: "top", ...o });
}
function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: fill }, objectName: name });
}
function chip(slide, x, y, w, label, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.21,
    fill: { color: fill }, line: { color: fill }, objectName: name });
  tb(slide, label, { x, y, w, h: 0.42, align: "center", valign: "middle", fontSize: 14,
    bold: true, color: C.background1 });
}
function numberDot(slide, x, y, label, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill },
    line: { color: fill } });
  tb(slide, label, { x, y, w: 0.42, h: 0.42, align: "center", valign: "middle",
    fontSize: 14, bold: true, color: C.background1 });
}

// 1. Title -----------------------------------------------------------------------------
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("Student Feedback Intelligence", { placeholder: "title" });
s.addText([
  { text: "A reusable aspect-based sentiment service for student feedback",
    options: { breakLine: true, fontSize: 18 } },
  { text: " ", options: { breakLine: true, fontSize: 10 } },
  { text: "Group 5  ·  E029 · E030 · E031 · E033 · E034 · E035", options: { breakLine: true } },
  { text: "SNLP Course Project 2026  ·  AI University", options: {} },
], { placeholder: "body" });
s.addNotes("Introduce the team and the one-line idea: we read student comments and say which part of the course each comment praises or criticises, with the words that show it.");

// 2. Problem ---------------------------------------------------------------------------
pres.addSection({ title: "Problem and data" });
s = content("Problem and data", "An average rating hides what students said");
card(s, 0.5, 1.25, 9.0, 1.15, HEX.lt2, "Example comment");
tb(s, "“The lectures were very informative, but the pace was too fast and the assignments were difficult.”",
  { x: 0.75, y: 1.38, w: 8.5, h: 0.9, fontSize: 18, italic: true, color: C.text2, valign: "middle" });
tb(s, "A single label (positive or negative) is wrong either way. What a coordinator needs:",
  { x: 0.5, y: 2.65, w: 9.0, h: 0.4, fontSize: 15 });
chip(s, 0.5, 3.2, 2.7, "teaching: positive", HEX.accent2, "chip teaching");
chip(s, 3.4, 3.2, 2.5, "pace: negative", HEX.accent3, "chip pace");
chip(s, 6.1, 3.2, 3.0, "assessment: negative", HEX.accent3, "chip assessment");
tb(s, "Our asset returns exactly this for every comment (the evidence words and a calibrated confidence for each), and turns a whole course into strengths, concerns and quotes.",
  { x: 0.5, y: 3.95, w: 9.0, h: 0.8, fontSize: 14, color: C.accent5 });
s.addNotes("Universities collect written feedback but faculty see averages. Read the handout's own example. Point out that our system's real output for it is exactly these three chips.");

// 3. Use case --------------------------------------------------------------------------
s = content("Problem and data", "Who uses it");
const users = [
  ["Course coordinator", "Course report: top strengths, top and emerging concerns, verbatim quotes"],
  ["Faculty member", "Which aspects of their course drew criticism, in the students' own words"],
  ["Department head", "Workload and pace concerns across a course, with caveats on small samples"],
  ["Group 6: Early Warning", "Course-level concern signals, advisory only, never about an individual"],
];
users.forEach(([who, what], i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.3 + Math.floor(i / 2) * 1.85;
  card(s, x, y, 4.4, 1.6, i === 3 ? "F6EBD9" : HEX.lt2, "user " + who);
  numberDot(s, x + 0.25, y + 0.25, String(i + 1), i === 3 ? HEX.accent4 : HEX.accent1);
  tb(s, who, { x: x + 0.85, y: y + 0.25, w: 3.4, h: 0.42, fontSize: 17, bold: true,
    color: C.text2, valign: "middle" });
  tb(s, what, { x: x + 0.85, y: y + 0.75, w: 3.35, h: 0.75, fontSize: 14 });
});
s.addNotes("Four users. Stress the fourth: Group 6 is the integration point the handout names (Group 5 to Group 6). Outputs are course-level and advisory for everyone.");

// 4. Data ------------------------------------------------------------------------------
s = content("Problem and data", "Data: 5,998 real, expert-labelled reviews");
const stats = [["5,998", "comments: course and teacher reviews"], ["8", "aspects"],
  ["4", "polarities incl. neutral, conflict"], ["4,198 / 900 / 900", "train / val / test, frozen"]];
stats.forEach(([big, small], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.25, 2.1, 1.55, HEX.lt2, "stat " + small);
  tb(s, big, { x: x + 0.1, y: 1.35, w: 1.9, h: 0.75, fontSize: i === 3 ? 16 : 34, bold: true,
    color: C.text2, align: "center", valign: "middle", fontFace: THEME.headFontFace });
  tb(s, small, { x: x + 0.1, y: 2.1, w: 1.9, h: 0.6, fontSize: 12, align: "center",
    color: C.accent5 });
});
tb(s, [
  { text: "EduRABSA (Hua et al., 2025), CC BY-NC-SA 4.0: Waterloo course reviews and RateMyProfessor teacher reviews, mapped from 24 categories to our 8 by written rules", options: { bullet: true, breakLine: true } },
  { text: "Why not write our own? A model scored on text the same six people wrote and labelled learns their phrasing, not the task", options: { bullet: true, breakLine: true } },
  { text: "Near-duplicates kept in one split; test split never used for any decision; every opening logged", options: { bullet: true } },
], { x: 0.5, y: 3.05, w: 9.0, h: 1.9, fontSize: 14, paraSpaceAfter: 6 });
s.addNotes("Public, licensed, expert-annotated. 11,271 aspect labels. Pace is derived by a rule with a full audit file. Biggest data caveat: North American and UK review sites, English only.");

// 5. NLP approach ----------------------------------------------------------------------
pres.addSection({ title: "Method and results" });
s = content("Method and results", "NLP approach: one attention query per aspect");
const steps = [
  ["Baseline", "TF-IDF word + char n-grams, logistic regression; aspect first, then polarity", HEX.accent5],
  ["Comparison", "Frozen MiniLM sentence embeddings + logistic regression", HEX.accent5],
  ["Our model", "DeBERTa-v3-base; 8 learned aspect queries, each reads its own words; 5-way head per aspect", HEX.accent1],
  ["Post-processing", "Thresholds and calibration tuned on validation only; evidence span grown to its clause", HEX.accent2],
];
steps.forEach(([name, what, fill], i) => {
  const y = 1.25 + i * 0.95;
  card(s, 0.5, y, 2.2, 0.78, fill, "step " + name);
  tb(s, name, { x: 0.5, y, w: 2.2, h: 0.78, fontSize: 16, bold: true, color: C.background1,
    align: "center", valign: "middle" });
  tb(s, what, { x: 2.95, y: y + 0.05, w: 6.5, h: 0.7, fontSize: 14, valign: "middle" });
});
s.addNotes("Baseline's biggest error was copying one tone onto every aspect of a mixed review, because it reads the whole comment. Our model gives each aspect its own attention, so 'teaching' reads 'lectures were informative' while 'pace' reads 'too fast'. Evidence supervision teaches the attention where to look. Trained on a 4 GB laptop GPU.");

// 6. Architecture ----------------------------------------------------------------------
s = content("Method and results", "Architecture");
const stages = [
  ["Input", "one comment, or up to 500 for a course (JSON)", HEX.lt2, C.text2],
  ["Preprocessing", "validation; PII redaction; casing kept", HEX.lt2, C.text2],
  ["NLP model", "DeBERTa-v3 + 8 aspect attention queries", HEX.accent1, C.background1],
  ["Post-processing", "frozen thresholds, calibration, evidence spans, aggregation", HEX.lt2, C.text2],
  ["API / asset", "FastAPI service, 6 endpoints, /docs", HEX.accent2, C.background1],
];
const sw = 1.6, sg = 0.25;
stages.forEach(([t, d, fill, ink], i) => {
  const x = 0.5 + i * (sw + sg);
  card(s, x, 1.3, sw, 1.95, fill, "stage " + t);
  tb(s, t, { x: x + 0.1, y: 1.4, w: sw - 0.2, h: 0.5, fontSize: 14, bold: true, color: ink,
    align: "center", valign: "middle" });
  tb(s, d, { x: x + 0.1, y: 1.95, w: sw - 0.2, h: 1.2, fontSize: 12, color: ink, align: "center" });
  if (i < stages.length - 1)
    s.addShape(pres.shapes.RIGHT_ARROW, { x: x + sw + 0.03, y: 2.15, w: sg - 0.06, h: 0.25,
      fill: { color: HEX.accent5 }, line: { color: HEX.accent5 } });
});
s.addShape(pres.shapes.DOWN_ARROW, { x: 7.95, y: 3.33, w: 0.3, h: 0.35,
  fill: { color: HEX.accent5 }, line: { color: HEX.accent5 } });
const apps = [["Course coordinator", "course report with quotes and caveats", HEX.lt2],
  ["Group 6: Early Warning", "course-level signals, advisory only", "F6EBD9"],
  ["Any other application", "HTTP + JSON, standard-library client", HEX.lt2]];
apps.forEach(([t, d, fill], i) => {
  const x = 0.5 + i * 3.07;
  card(s, x, 3.75, 2.86, 1.25, fill, "app " + t);
  tb(s, t, { x: x + 0.15, y: 3.85, w: 2.56, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  tb(s, d, { x: x + 0.15, y: 4.3, w: 2.56, h: 0.6, fontSize: 12, color: C.accent5 });
});
s.addNotes("Walk top to bottom: input, redaction, model, frozen decisions and calibration, the FastAPI service, and the three consumers. All NLP lives in pipeline.py; the API only adds HTTP, so module and API give identical results.");

// 7. Results ---------------------------------------------------------------------------
s = content("Method and results", "Results: +0.252 tuple-F1 over the baseline");
s.addChart(pres.charts.BAR, [{ name: "Tuple-F1 (test, 900 comments)",
  labels: ["Majority class", "TF-IDF baseline", "MiniLM embeddings", "Our model"],
  values: [0.216, 0.452, 0.500, 0.704] }], {
  x: 0.4, y: 1.15, w: 5.6, h: 3.85, barDir: "bar", chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12,
  dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt",
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 0.85, valGridLine: { style: "none" },
  catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Exact-match tuple-F1 on the test split", titleFontSize: 13, titleColor: HEX.dk2,
  titleFontFace: "+mn-lt", barGapWidthPct: 60, catAxisOrientation: "maxMin", objectName: "results chart" });
const kpis = [["0.704", "tuple-F1, 95% CI [0.683, 0.723]"], ["0.856", "aspect detection F1"],
  ["0.822", "polarity accuracy on found aspects"]];
kpis.forEach(([big, small], i) => {
  const y = 1.2 + i * 1.28;
  card(s, 6.3, y, 3.2, 1.1, HEX.lt2, "kpi " + small);
  tb(s, big, { x: 6.45, y: y + 0.08, w: 2.9, h: 0.55, fontSize: 28, bold: true, color: C.text2,
    fontFace: THEME.headFontFace });
  tb(s, small, { x: 6.45, y: y + 0.62, w: 2.9, h: 0.4, fontSize: 12, color: C.accent5 });
});
s.addNotes("Paired bootstrap difference against the baseline: +0.252, CI +0.225 to +0.279. Pretraining alone is not the reason: frozen embeddings add only 0.048. Calibration: document-sentiment ECE 0.116 to 0.021. Weakest: neutral and conflict polarities; per aspect, resources 0.487.");

// 8. Demonstration ---------------------------------------------------------------------
pres.addSection({ title: "Demo and reuse" });
s = content("Demo and reuse", "Live demonstration");
const demo = [
  ["Start the service", "python src\\api.py, then open 127.0.0.1:8000/docs"],
  ["The handout example", "teaching +, pace −, assessment −, with evidence words and calibrated scores"],
  ["A whole course", "150 real comments: strengths, top concern workload, emerging concern pace"],
  ["Another group calls it", "examples/client_example.py: a separate program invoking our API over HTTP"],
  ["Group 6 signals", "POST /v1/support-signals: course-level, advisory flags, no comments"],
];
demo.forEach(([t, d], i) => {
  const y = 1.15 + i * 0.63;
  numberDot(s, 0.5, y + 0.06, String(i + 1), HEX.accent1);
  tb(s, t, { x: 1.1, y, w: 2.6, h: 0.55, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
  tb(s, d, { x: 3.7, y, w: 5.8, h: 0.55, fontSize: 14, valign: "middle" });
});
card(s, 0.5, 4.38, 9.0, 0.72, "FBE6DF", "failure case");
tb(s, [
  { text: "Failure, on purpose:  ", options: { bold: true, color: C.accent3 } },
  { text: "“Oh great, another surprise quiz on Monday.” → nothing found. Sarcasm is read literally (0 of 4 probes)." },
], { x: 0.7, y: 4.38, w: 8.6, h: 0.72, fontSize: 14, valign: "middle" });
s.addNotes("Follow docs/Group5_Demo_Script. Use the /docs page dropdown examples. Step 5 is the failure: sarcasm, then optionally Hinglish ('Sir ka padhane ka tareeka bahut accha hai' finds nothing). Explain why: neither appears in the English review training data.");

// 9. Reusable asset --------------------------------------------------------------------
s = content("Demo and reuse", "Reusable asset: what another group needs");
card(s, 0.5, 1.2, 4.3, 3.85, HEX.lt2, "endpoints");
tb(s, "HTTP + JSON  (FastAPI)", { x: 0.75, y: 1.35, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
tb(s, [
  { text: "POST /v1/analyze-feedback", options: { breakLine: true } },
  { text: "POST /v1/analyze-feedback/batch", options: { breakLine: true } },
  { text: "POST /v1/support-signals", options: { breakLine: true } },
  { text: "GET /v1/schema /health /model-card", options: {} },
], { x: 0.75, y: 1.85, w: 3.9, h: 1.5, fontSize: 13, fontFace: "Courier New", color: C.text2, paraSpaceAfter: 4 });
tb(s, "Typed schemas → interactive docs at /docs; one error format; standard-library client in examples/; README answers the handout's reusable-asset test.",
  { x: 0.75, y: 3.45, w: 3.9, h: 1.4, fontSize: 13, color: C.accent5 });
card(s, 5.1, 1.2, 4.4, 3.85, "F6EBD9", "group 6");
tb(s, "Integration with Group 6", { x: 5.35, y: 1.35, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
tb(s, [
  { text: "We serve course-level concern signals with advisory_only and not_for_individual_action flags", options: { bullet: true, breakLine: true } },
  { text: "We built 38 comments in their own JSON format: train split only, names excluded, 8 marked synthetic", options: { bullet: true, breakLine: true } },
  { text: "Their reply is read back as counts only; shown today with a clearly labelled stand-in", options: { bullet: true } },
], { x: 5.35, y: 1.85, w: 4.0, h: 3.1, fontSize: 13, paraSpaceAfter: 6 });
s.addNotes("This is deliverable E plus the integration marks. If Group 6's real reply has arrived, say so and show the merged counts; if not, say honestly that the stand-in tests our plumbing only.");

// 10. Limitations & future work ---------------------------------------------------------
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK_TOP", sectionTitle: "Close" });
s.addText("Limitations and future work", { placeholder: "title" });
tb(s, [
  { text: "What it does not do well", options: { bold: true, color: C.background1, breakLine: true } },
  { text: "Neutral and mixed opinions: 31 of 116 and 56 of 148 recognised", options: { bullet: true, breakLine: true } },
  { text: "Sarcasm and Hinglish: 0 of 4 and 0 of 5 probes", options: { bullet: true, breakLine: true } },
  { text: "Blames the teacher: spurious teaching: negative in 5 of 15 failed probes", options: { bullet: true, breakLine: true } },
  { text: "Untested on NMIMS feedback; bare names not redacted", options: { bullet: true } },
], { x: 0.6, y: 1.7, w: 4.3, h: 3.2, fontSize: 15, color: C.accent6, paraSpaceAfter: 8 });
tb(s, [
  { text: "To make it production-ready", options: { bold: true, color: C.background1, breakLine: true } },
  { text: "Label a few hundred NMIMS comments, including Hinglish", options: { bullet: true, breakLine: true } },
  { text: "Named-entity redaction before any real data", options: { bullet: true, breakLine: true } },
  { text: "Bias probes before every release; class-weighted training for neutral and conflict", options: { bullet: true, breakLine: true } },
  { text: "Compare semesters and courses (stretch goal); live Group 6 link", options: { bullet: true } },
], { x: 5.1, y: 1.7, w: 4.3, h: 3.2, fontSize: 15, color: C.accent6, paraSpaceAfter: 8 });
s.addNotes("End on honesty: the handout rewards recognising limits. Then thank the audience and take questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, "slides:", n + 2);
})();
