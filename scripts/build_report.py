"""Markdown report -> .docx (+ .pdf) in the house style of the Week 1-2 report.

    python scripts/build_report.py docs/Group5_Week3_Report.md            # docx + pdf
    python scripts/build_report.py docs/Group5_Week3_Report.md --no-pdf

The .md is the source; the .docx and .pdf are generated and never edited by hand.
Formatting is copied from the Week 1-2 v2 report (Calibri, navy headings, shaded table
headers, A4 with the same margins) so every weekly report looks like one series.

Supported markdown: a front-matter block for the title page, ## and ### headings,
paragraphs, **bold**, *italic*, `code`, - bullets, 1. numbered items, pipe tables,
``` fenced blocks, > notes, ![caption](image.png) and <!-- pagebreak -->.
Everything before the first ## heading is the markdown reader's header and is skipped
(the title page comes from the front matter).

The PDF is exported through Word and must be e-mail safe: no link annotations and no
emoji fonts (Gmail flags PDFs with many live links). The script checks both.
"""

import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu

NAVY, BLUE, GREY, DARK, RED = "1F3864", "2E5C8A", "595959", "333333", "C00000"
HEADER_FILL = "DEE6F0"
TABLE_WIDTH = 9400          # dxa, as in the Week 1-2 report
BODY_PT2 = 19               # half-points: 9.5 pt body text


# --- low-level XML helpers ------------------------------------------------------

def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def _rpr(bold=False, italic=False, color=None, size=BODY_PT2, font="Calibri"):
    r = _el("w:rPr")
    r.append(_el("w:rFonts", **{"w:ascii": font, "w:hAnsi": font, "w:cs": font,
                                "w:eastAsia": font}))
    if bold:
        r.append(_el("w:b"))
        r.append(_el("w:bCs"))
    if italic:
        r.append(_el("w:i"))
        r.append(_el("w:iCs"))
    if color:
        r.append(_el("w:color", **{"w:val": color}))
    r.append(_el("w:sz", **{"w:val": size}))
    r.append(_el("w:szCs", **{"w:val": size}))
    return r


def _run(text, **fmt):
    r = _el("w:r")
    r.append(_rpr(**fmt))
    t = _el("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def _ppr(style=None, before=None, after=None, line=None, jc=None, ind_left=None,
         hanging=None, border_bottom=None, borders_tb=None, keep_next=False, numid=None):
    p = _el("w:pPr")
    if style:
        p.append(_el("w:pStyle", **{"w:val": style}))
    if keep_next:
        p.append(_el("w:keepNext"))
    if numid is not None:
        n = _el("w:numPr")
        n.append(_el("w:ilvl", **{"w:val": 0}))
        n.append(_el("w:numId", **{"w:val": numid}))
        p.append(n)
    if border_bottom or borders_tb:
        b = _el("w:pBdr")
        if borders_tb:
            b.append(_el("w:top", **{"w:val": "single", "w:color": borders_tb, "w:sz": 4,
                                     "w:space": 10}))
            b.append(_el("w:bottom", **{"w:val": "single", "w:color": borders_tb, "w:sz": 4,
                                        "w:space": 10}))
        else:
            b.append(_el("w:bottom", **{"w:val": "single", "w:color": border_bottom,
                                        "w:sz": 4, "w:space": 4}))
        p.append(b)
    sp = {}
    if before is not None:
        sp["w:before"] = before
    if after is not None:
        sp["w:after"] = after
    if line is not None:
        sp["w:line"] = line
        sp["w:lineRule"] = "auto"
    if sp:
        p.append(_el("w:spacing", **sp))
    if ind_left is not None:
        ind = {"w:left": ind_left}
        if hanging:
            ind["w:hanging"] = hanging
        p.append(_el("w:ind", **ind))
    if jc:
        p.append(_el("w:jc", **{"w:val": jc}))
    return p


INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def _inline_runs(text, size=BODY_PT2, color=None, bold=False, italic=False):
    runs = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            runs.append(_run(part[2:-2], bold=True, italic=italic, size=size, color=color))
        elif part.startswith("`") and part.endswith("`"):
            runs.append(_run(part[1:-1], font="Consolas", size=size - 1, color=DARK,
                             bold=bold, italic=italic))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs.append(_run(part[1:-1], italic=True, bold=bold, size=size, color=color))
        else:
            runs.append(_run(part, bold=bold, italic=italic, size=size, color=color))
    return runs


class Builder:
    def __init__(self, template):
        self.doc = Document(template)
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self.body = body
        self.sectPr = body.find(qn("w:sectPr"))

    def _add(self, el):
        self.sectPr.addprevious(el)
        return el

    def para(self, ppr, runs):
        p = _el("w:p")
        p.append(ppr)
        for r in runs:
            p.append(r)
        return self._add(p)

    # --- blocks -----------------------------------------------------------

    def title_page(self, fm):
        self.para(_ppr(before=1400, after=60, jc="center"),
                  [_run(fm["title"], bold=True, color=NAVY, size=40)])
        self.para(_ppr(after=30, jc="center"), [_run(fm.get("subtitle", ""), color=GREY, size=22)])
        self.para(_ppr(after=420, jc="center"), [_run(fm.get("week", ""), bold=True, size=21)])
        self.para(_ppr(after=40, jc="center", borders_tb=NAVY),
                  [_run(fm.get("course", ""), bold=True, size=21)])
        self.para(_ppr(before=120, after=900, jc="center"),
                  [_run(fm.get("members", ""), color=GREY, size=19)])
        self.para(_ppr(after=30, jc="center"),
                  [_run(fm.get("version", ""), italic=True, color=GREY, size=17)])
        if fm.get("note"):
            self.para(_ppr(jc="center"), [_run(fm["note"], italic=True,
                                               color=fm.get("note_color", GREY), size=17)])
        self.page_break()

    def page_break(self):
        p = _el("w:p")
        r = _el("w:r")
        r.append(_el("w:br", **{"w:type": "page"}))
        p.append(r)
        self._add(p)

    def h1(self, text):
        self.para(_ppr(style="Heading1", before=280, after=110, border_bottom="BFBFBF",
                       keep_next=True), [_run(text, bold=True, color=NAVY, size=24)])

    def h2(self, text):
        self.para(_ppr(style="Heading2", before=170, after=70, keep_next=True),
                  [_run(text, bold=True, color=BLUE, size=20)])

    def body_text(self, text):
        self.para(_ppr(after=110, line=250, jc="both"), _inline_runs(text))

    def note(self, text):
        self.para(_ppr(after=110, line=250, ind_left=240), _inline_runs(text, italic=True,
                                                                        color=GREY, size=18))

    def bullet(self, text):
        self.para(_ppr(style="ListParagraph", numid=1, after=55, line=240),
                  _inline_runs(text))

    def numbered(self, num, text):
        self.para(_ppr(after=55, line=240, ind_left=720, hanging=360),
                  [_run(f"{num}.\t")] + _inline_runs(text))

    def mono(self, lines):
        for i, ln in enumerate(lines):
            self.para(_ppr(after=30 if i < len(lines) - 1 else 110, ind_left=240),
                      [_run(ln if ln else " ", font="Consolas", color=DARK, size=17)])

    def image(self, path, caption):
        p = self.doc.add_paragraph()
        p.alignment = 1
        p.add_run().add_picture(str(path), width=Cm(16))
        self.sectPr.addprevious(p._p)
        if caption:
            self.para(_ppr(after=110, jc="center"), [_run(caption, italic=True, color=GREY,
                                                          size=17)])

    def table(self, rows):
        ncol = len(rows[0])
        # Every column gets at least its longest single word (so headers never break
        # mid-word); the remaining width goes to columns in proportion to how much
        # longer their content is than that minimum.
        def cell(r, c):
            return re.sub(r"[*`]", "", r[c]) if c < len(r) else ""
        min_ch = [max(len(w) for r in rows for w in (cell(r, c).split() or [""])) + 2
                  for c in range(ncol)]
        want_ch = [max(min(len(cell(r, c)), 70) for r in rows) + 2 for c in range(ncol)]
        per_ch = 105                                   # dxa per character at 8.5 pt
        base = [m * per_ch for m in min_ch]
        spare = TABLE_WIDTH - sum(base)
        extra = [max(0, w - m) for w, m in zip(want_ch, min_ch)]
        if spare > 0 and sum(extra):
            widths = [int(b + spare * e / sum(extra)) for b, e in zip(base, extra)]
        else:
            widths = [int(TABLE_WIDTH * w / sum(want_ch)) for w in want_ch]
        widths[-1] += TABLE_WIDTH - sum(widths)
        tbl = _el("w:tbl")
        pr = _el("w:tblPr")
        pr.append(_el("w:tblW", **{"w:type": "dxa", "w:w": TABLE_WIDTH}))
        b = _el("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b.append(_el(f"w:{side}", **{"w:val": "single", "w:color": "auto", "w:sz": 4}))
        pr.append(b)
        pr.append(_el("w:tblLayout", **{"w:type": "fixed"}))
        tbl.append(pr)
        grid = _el("w:tblGrid")
        for w in widths:
            grid.append(_el("w:gridCol", **{"w:w": w}))
        tbl.append(grid)
        for ri, row in enumerate(rows):
            tr = _el("w:tr")
            if ri == 0:
                trpr = _el("w:trPr")
                trpr.append(_el("w:tblHeader"))
                tr.append(trpr)
            else:
                trpr = _el("w:trPr")
                trpr.append(_el("w:cantSplit"))
                tr.append(trpr)
            for ci in range(ncol):
                txt = row[ci] if ci < len(row) else ""
                tc = _el("w:tc")
                tcpr = _el("w:tcPr")
                tcpr.append(_el("w:tcW", **{"w:type": "dxa", "w:w": widths[ci]}))
                if ri == 0:
                    tcpr.append(_el("w:shd", **{"w:fill": HEADER_FILL, "w:color": "auto",
                                                "w:val": "clear"}))
                mar = _el("w:tcMar")
                for side, v in (("top", 60), ("left", 90), ("bottom", 60), ("right", 90)):
                    mar.append(_el(f"w:{side}", **{"w:type": "dxa", "w:w": v}))
                tcpr.append(mar)
                tc.append(tcpr)
                p = _el("w:p")
                p.append(_ppr(after=0, line=230))
                for r in _inline_runs(txt, size=17, bold=(ri == 0)):
                    p.append(r)
                tc.append(p)
                tr.append(tc)
            tbl.append(tr)
        self._add(tbl)
        self.para(_ppr(after=60), [])

    def footer(self, text):
        ftr = self.doc.sections[0].footer
        p = ftr.paragraphs[0]
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        for child in list(p._p):
            if child.tag == qn("w:r"):
                p._p.remove(child)
        p._p.append(_run(text.rstrip() + "  ", color="808080", size=15))
        r = _el("w:r")
        r.append(_rpr(color="808080", size=15))
        r.append(_el("w:fldChar", **{"w:fldCharType": "begin"}))
        it = _el("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = "PAGE"
        r.append(it)
        r.append(_el("w:fldChar", **{"w:fldCharType": "separate"}))
        r.append(_el("w:fldChar", **{"w:fldCharType": "end"}))
        p._p.append(r)

    def save(self, path, fm):
        cp = self.doc.core_properties
        cp.title = f"{fm['title']} - {fm.get('week', '')}"
        cp.author = "Group 5 (E029, E030, E031, E033, E034, E035)"
        cp.comments = ""
        self.doc.save(path)


# --- markdown parsing -----------------------------------------------------------

def parse_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit("report markdown needs a --- front matter block")
    fm = {}
    for ln in m.group(1).splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            fm[k.strip()] = v.strip().strip('"')
    return fm, text[m.end():]


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build(md_path, out_docx, template):
    fm, body = parse_front_matter(md_path.read_text(encoding="utf-8"))
    b = Builder(template)
    b.title_page(fm)
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("## "):
        i += 1                                  # skip the markdown-only header
    para = []

    def flush():
        if para:
            b.body_text(" ".join(s.strip() for s in para))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            b.mono(block)
        elif s == "<!-- pagebreak -->":
            flush()
            b.page_break()
        elif s.startswith("## "):
            flush()
            b.h1(s[3:].strip())
        elif s.startswith("### "):
            flush()
            b.h2(s[4:].strip())
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in r):
                    rows.append(r)
                i += 1
            b.table(rows)
            continue
        elif s.startswith("- "):
            flush()
            item = s[2:]
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip():
                i += 1
                item += " " + lines[i].strip()
            b.bullet(item)
        elif re.match(r"^\d+\. ", s):
            flush()
            num, item = s.split(". ", 1)
            while i + 1 < len(lines) and lines[i + 1].startswith("   ") and lines[i + 1].strip():
                i += 1
                item += " " + lines[i].strip()
            b.numbered(num, item)
        elif s.startswith("> "):
            flush()
            note = s[2:]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("> "):
                i += 1
                note += " " + lines[i].strip()[2:]
            b.note(note)
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            b.image((md_path.parent / m.group(2)).resolve(), m.group(1))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    b.footer(fm.get("footer", ""))
    b.save(out_docx, fm)
    return out_docx


def export_pdf(docx_path, pdf_path):
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(docx_path.resolve()), ReadOnly=True)
        d.Fields.Update()
        d.ExportAsFixedFormat(str(pdf_path.resolve()), 17)      # 17 = wdExportFormatPDF
        d.Close(False)
    finally:
        word.Quit()
    import fitz
    doc = fitz.open(str(pdf_path))
    links = sum(len(p.get_links()) for p in doc)
    fonts = {f[3] for p in doc for f in p.get_fonts()}
    emoji = [f for f in fonts if "Emoji" in f]
    n = doc.page_count
    doc.close()
    if links or emoji:
        raise SystemExit(f"PDF not e-mail safe: {links} links, emoji fonts {emoji}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("md", type=Path)
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).resolve().parent / "report_template.docx")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    out = a.md.with_suffix(".docx")
    build(a.md, out, a.template)
    print(f"docx -> {out}")
    if not a.no_pdf:
        pages = export_pdf(out, a.md.with_suffix(".pdf"))
        print(f"pdf  -> {a.md.with_suffix('.pdf')} ({pages} pages, 0 links, no emoji fonts)")
