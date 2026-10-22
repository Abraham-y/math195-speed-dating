"""Two implementation slides for the restructured midterm deck: one after "BIC and Stepwise Logistic Regression" and one
after "LDA and QDA". They are built by cloning the styling (cards, headings, result strip, footer, background) of the
deck's own slides 19 and 23, so they match when imported into Google Slides (File > Import slides).

Input : the team deck downloaded as .pptx (--template; only its styling is used, none of its slides are kept);
        report/figures/slide_stepwise_kept.png, slide_qda_shrink.png, slide_eq_bic.png, slide_eq_shrink.png
        (scripts/report_figures.py --slides)
Output: results/report_impl_slides.pptx: the two slides with speaker notes;
        with --full, results/report_impl_slides_full.pptx: the whole deck with the two slides in place of its empty
        slides 14 and 17, the tuning slide moved into Results, four slide errors fixed (main result 0.54 -> 0.58,
        QDA covariance "2775" -> "about 8,000" numbers, the stock ROC image -> our curves, the pre-registration footer)
        and footers renumbered. The team's speaker notes are kept exactly as they were.
Numbers on the slides: AUCs from results/part1_models_metrics.csv (headline, event 5-fold CV); column counts, the QDA
shrinkage chosen and the training sizes from results/part1_models_folds.csv; predictors kept from
results/part1_models_stepwise_selection.csv. Nothing is random.
Run from the repository root: python scripts/report_impl_slides.py --template "path/to/Speed-dating midterm presentation.pptx" [--full]
"""
import argparse, copy, io, pathlib, re
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.ns import qn

RES, FIG = pathlib.Path("results"), pathlib.Path("report/figures")


def find(slide, start):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith(start):
            return sh
    raise KeyError(start)


def clone(shape, slide, left, top, width, height):
    el = copy.deepcopy(shape._element)
    tree = slide.shapes._spTree
    ext = tree.find(qn("p:extLst"))
    ext.addprevious(el) if ext is not None else tree.append(el)
    sh = slide.shapes[-1] if ext is None else next(s for s in slide.shapes if s._element is el)
    sh.left, sh.top, sh.width, sh.height = Inches(left), Inches(top), Inches(width), Inches(height)
    return sh


def set_text(sh, paras):
    """Replace the text, keeping the first paragraph's and first run's formatting. paras: strings or lists of (text, bold)."""
    body = sh.text_frame._txBody
    ps = body.findall(qn("a:p")); proto = ps[0]
    for p in ps:
        body.remove(p)
    for para in paras:
        runs = [(para, None)] if isinstance(para, str) else para
        p = copy.deepcopy(proto); rs = p.findall(qn("a:r")); rproto = rs[0]
        for e in rs + p.findall(qn("a:br")) + p.findall(qn("a:fld")):
            p.remove(e)
        end = p.find(qn("a:endParaRPr"))
        for text, bold in runs:
            r = copy.deepcopy(rproto); r.find(qn("a:t")).text = text
            if bold is not None:
                rpr = r.find(qn("a:rPr"))
                if rpr is None:
                    rpr = r.makeelement(qn("a:rPr"), {}); r.insert(0, rpr)
                rpr.set("b", "1" if bold else "0")
            end.addprevious(r) if end is not None else p.append(r)
        body.append(p)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = " ".join(text.split())


def new_slide(prs, tmpl):
    s = prs.slides.add_slide(tmpl.slide_layout)
    bg = tmpl._element.cSld.find(qn("p:bg"))
    if bg is not None:
        s._element.cSld.insert(0, copy.deepcopy(bg))
    return s


def slide_with(prs, text):
    for s in prs.slides:
        if any(sh.has_text_frame and text in sh.text_frame.text for sh in s.shapes):
            return s
    raise KeyError(text)


def build(prs):
    """Append the two implementation slides to prs, styled from its own slides; returns them."""
    t19, t23 = slide_with(prs, "Hold out whole events"), slide_with(prs, "Too many predictors")
    eyebrow, title, footer = find(t19, "4 · TECHNICAL"), find(t19, "Hold out whole"), find(t19, "Math 195")
    card = next(sh for sh in t19.shapes if sh.has_text_frame and sh.text_frame.text == "" and sh.top < Inches(2.2))
    h3, body = find(t19, "Why"), find(t19, "People date up to")
    strip = next(sh for sh in t23.shapes if sh.has_text_frame and sh.text_frame.text == "" and sh.top > Inches(4.9))
    strip_text = find(t23, "Simpler beat fuller")
    icon = next(sh for sh in t23.shapes if sh.shape_type == 13)
    icon_blob = icon.image.blob

    def header(s, title_text, page, foot):
        set_text(clone(eyebrow, s, 0.89, 0.89, 11.9, 0.26), ["4 · TECHNICAL IMPLEMENTATION"])
        set_text(clone(title, s, 0.89, 1.29, 11.9, 0.52), [title_text])
        set_text(clone(footer, s, 0.89, 6.88, 11.9, 0.26), [f"Math 195 · Speed dating · {page} · {foot}"])

    def card_at(s, left, top, width, height, heading, text, text_h):
        clone(card, s, left, top, width, height)
        set_text(clone(h3, s, left + 0.2, top + 0.17, width - 0.4, 0.29), [heading])
        set_text(clone(body, s, left + 0.2, top + 0.5, width - 0.4, text_h), [text])

    def strip_at(s, left, top, width, height, runs):
        clone(strip, s, left, top, width, height)
        s.shapes.add_picture(io.BytesIO(icon_blob), Inches(left + 0.22), Inches(top + height / 2 - 0.16), height=Inches(0.33))
        set_text(clone(strip_text, s, left + 0.62, top + 0.1, width - 0.75, height - 0.2), [runs])

    # ---- Slide 14: logistic regression and stepwise selection on our data
    a = s = new_slide(prs, t19)
    header(s, "Stepwise logistic regression on our data", 14,
           "BIC: James et al. (2023), ISL §6.1.3 · Direction from the full logistic model · AUCs on held-out events (event 5-fold CV)")
    card_at(s, 0.9, 1.95, 4.95, 2.2, "The full model",
            "All 74 pre-date predictors, about 125 columns. Curves (splines) for age, age gap, interests and fit were tried, "
            "but BIC never chose one.", 0.85)
    s.shapes.add_picture(str(FIG / "slide_eq_bic.png"), Inches(1.1), Inches(3.5), width=Inches(3.5))
    card_at(s, 0.9, 4.3, 4.95, 1.3, "Stepwise selection by BIC",
            "Kept about 8 of the 74 predictors in each fold; about 15 of 90 with the date ratings.", 0.6)
    s.shapes.add_picture(str(FIG / "slide_stepwise_kept.png"), Inches(6.15), Inches(1.95), width=Inches(6.1))
    strip_at(s, 0.9, 5.85, 11.54, 0.8, [("Fewer predictors predicted better. ", True),
             ("Held-out AUC before the date: full model 0.542, stepwise 0.559. With the date ratings: 0.780 and 0.835.", False)])
    notes(s, """Here's how logistic regression ran on our data. The full model uses all 74 pre-date predictors, about 125
        columns once categories become yes-or-no dummy columns. We let BIC decide whether age, the age gap, interest
        similarity or preference fit should get a curve instead of a straight line. Each extra coefficient costs log n,
        about 8 with roughly 3,300 training pairs, and a curve never earned that, so the model stayed linear. Stepwise
        selection then used BIC to choose predictors, and it kept only about 8 of the 74. The chart shows which ones it
        kept most often, out of 15 fits: three directions in each of five folds. His self-rated fun, the age gap, how
        often he dates, sharing a field of study, and how much he values intelligence came up in most fits. Blue
        means more of it goes with more matches; orange, with fewer. And keeping fewer predictors predicted better on
        new events: 0.559 instead of 0.542 before the date, and 0.835 instead of 0.780 with the date ratings.""")

    # ---- Slide 17: LDA and QDA on our data
    b = s = new_slide(prs, t19)
    header(s, "LDA and QDA on our data", 17,
           "Regularized QDA: Friedman (1989); Hastie et al. (2009) §4.3.1 · Heatmaps: 8 of our predictors, 690 matched pairs")
    card_at(s, 0.9, 1.95, 4.95, 1.55, "LDA",
            "One covariance matrix shared by matches and non-matches. Prior: the training match rate, about 0.165. "
            "Nothing to tune.", 0.85)
    card_at(s, 0.9, 3.65, 4.95, 2.6, "QDA, regularized",
            "Its own covariance matrix for the matches: about 8,000 numbers from about 550 matches. So we shrink it "
            "toward “no correlation”:", 0.85)
    s.shapes.add_picture(str(FIG / "slide_eq_shrink.png"), Inches(1.1), Inches(5.08), height=Inches(0.42))
    set_text(clone(body, s, 1.1, 5.62, 4.55, 0.35), ["Inner CV picked r = 0.9 in all 10 fits."])
    s.shapes.add_picture(str(FIG / "slide_qda_shrink.png"), Inches(6.05), Inches(1.95), width=Inches(6.4))
    strip_at(s, 6.05, 5.3, 6.4, 0.95, [("Before the date: ", True),
             ("LDA 0.553, regularized QDA 0.579. With the date ratings: 0.790 and 0.836.", False)])
    notes(s, """Here's how LDA and QDA ran on our data. LDA estimates one covariance matrix, shared by matches and
        non-matches, and uses the training match rate, about 0.165, as the prior. It has nothing to tune. QDA estimates
        a separate covariance matrix for the matches. With about 125 columns that's about 8,000 numbers, estimated from
        only about 550 matches in a training fold, so many of them are noise. So we shrink it toward the identity
        matrix, which means no correlations: a weighted average with weight r. The heatmaps show what that does, for 8
        of our predictors among the matched pairs. On the left are the correlations we estimate; for example, women who
        rate themselves as more attractive also rate themselves as more fun, a correlation of 0.44. On the right is what
        QDA uses after shrinking with r equal to 0.9: every correlation is about a tenth as big, so 0.44 becomes 0.04.
        Our inner cross-validation picked 0.9, the strongest option, in all ten fits. And regularized QDA ended up one
        of our best models: 0.579 before the date, against 0.553 for LDA.""")

    return a, b


def replace_in_frame(tf, old, new):
    """Replace text in a text frame; a match split across runs is collapsed into the paragraph's first run."""
    hit = 0
    for p in tf.paragraphs:
        full = "".join(r.text for r in p.runs)
        if old not in full:
            continue
        one = next((r for r in p.runs if old in r.text), None)
        if one is not None:
            one.text = one.text.replace(old, new)
        else:
            p.runs[0].text = full.replace(old, new)
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
        hit += 1
    return hit


def replace_text(slide, old, new):
    hit = 0
    for sh in slide.shapes:
        if sh.has_text_frame:
            hit += replace_in_frame(sh.text_frame, old, new)
        if sh.has_table:
            hit += sum(replace_in_frame(c.text_frame, old, new) for c in sh.table.iter_cells())
    assert hit, f"{old!r} not found"


def replace_image(slide, path):
    """Point the slide's (only) picture at a new image, keeping its height and centre."""
    from PIL import Image
    pic = next(sh for sh in slide.shapes if sh.shape_type == 13)
    _, rid = slide.part.get_or_add_image_part(str(path))
    fill = pic._element.blipFill
    fill.find(qn("a:blip")).set(qn("r:embed"), rid)
    crop = fill.find(qn("a:srcRect"))
    if crop is not None:
        fill.remove(crop)
    w, h = Image.open(path).size; cx = pic.left + pic.width // 2
    pic.width = int(pic.height * w / h); pic.left = int(cx - pic.width // 2)


def assemble(prs, a, b):
    """The whole deck: the new slides replace the two empty ones, the tuning slide moves into Results after
    "Every model is close to chance", four slide errors are fixed and the footers renumbered. Speaker notes of the
    team's slides are left exactly as they were."""
    ids = prs.slides._sldIdLst; slides = list(prs.slides); el = {int(e.get("id")): e for e in ids}
    new_ids = (a.slide_id, b.slide_id)
    blanks = [s for s in slides if s.slide_id not in new_ids and not any(sh.has_text_frame and sh.text_frame.text.strip() or sh.shape_type == 13 for sh in s.shapes)]
    assert len(blanks) == 2, f"expected two empty slides, found {len(blanks)}"
    swap = {blanks[0].slide_id: a, blanks[1].slide_id: b}
    order = [swap.get(s.slide_id, s) for s in slides if s.slide_id not in new_ids]
    pos = lambda sl: next(i for i, s in enumerate(order) if s.slide_id == sl.slide_id)
    tuning, chance = slide_with(prs, "Too many predictors"), slide_with(prs, "Every model is close to chance")
    t = order.pop(pos(tuning)); order.insert(pos(chance) + 1, t)
    blank_ids, order_ids = [s.slide_id for s in blanks], [s.slide_id for s in order]     # before any change
    for i in blank_ids:
        prs.part.drop_rel(el[i].get(qn("r:id")))
    for e in list(ids):
        ids.remove(e)
    for i in order_ids:
        ids.append(el[i])
    replace_text(tuning, "4 · WHAT THE TUNING TOLD US", "5 · RESULTS")
    main_result = slide_with(prs, "THE MAIN RESULT")
    big = [sh for sh in main_result.shapes if sh.has_text_frame and sh.text_frame.text.strip() == "0.54"]
    assert len(big) == 1; replace_in_frame(big[0].text_frame, "0.54", "0.58")      # best pre-date AUC is 0.583
    replace_text(slide_with(prs, "Methods we used beyond"), "(2775 numbers)", "(about 8,000 numbers)")   # ~125 columns
    replace_text(slide_with(prs, "The same pre-processing pipeline"), " · Pre-registration: Nosek et al. (2018), PNAS", "")
    cv_slide = slide_with(prs, "Hold out whole events")
    replace_text(slide_with(prs, "Our folds are whole events"), "slide 17", f"slide {pos(cv_slide) + 1}")
    replace_image(slide_with(prs, "Model evaluation: the ROC curve"), FIG / "slide_roc.png")       # our curves, not a stock image
    replace_image(cv_slide, FIG / "slide_cv_design.png")                                           # training set / test set
    for i, s in enumerate(order, 1):
        for sh in s.shapes:
            m = sh.has_text_frame and re.match(r"Math 195 · Speed dating · (\d+)", sh.text_frame.text)
            if m:
                replace_in_frame(sh.text_frame, m.group(0), f"Math 195 · Speed dating · {i}")
    return order


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--template", required=True)
    ap.add_argument("--full", action="store_true", help="write the whole deck with the new slides in place")
    args = ap.parse_args()
    prs = Presentation(args.template); n_old = len(prs.slides)
    a, b = build(prs)
    if args.full:
        order = assemble(prs, a, b); out = RES / "report_impl_slides_full.pptx"
    else:
        ids = prs.slides._sldIdLst
        for sid in list(ids)[:n_old]:
            prs.part.drop_rel(sid.get(qn("r:id"))); ids.remove(sid)
        out = RES / "report_impl_slides.pptx"
    prs.save(out); print(f"wrote {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
