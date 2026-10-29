"""Fixes for the team's midterm deck once it already holds Abraham's implementation slides (the Google Slides version
exported as .pptx), then a page-by-page check of the slides that will be shown.

Input : the team deck as .pptx (--deck)
Output: results/report_deck_fixed.pptx, and the check printed (page, title, footer page, pictures replaced).
The talk does not cover the random forest, so the shown slides quote the best of the other Part 1 models
(results/part1_models_metrics.csv, headline spec, event 5-fold): regularized QDA before the date (0.579, SE 0.023) and
the lasso with the date ratings (0.838, SE 0.008; QDA 0.836).
Input figures: report/figures/slide_roc_qda.png (scripts/report_roc.py --model qda), slide_cv_design.png,
slide_auc_part1_talk.png (scripts/report_figures.py --slides).
Fixes, each asserted to apply exactly where expected:
  - main result 0.54 -> 0.58 and 0.85 -> 0.84, footer: QDA before, lasso after (0.54 is the full logistic model only);
  - three images: the stock ROC curve -> QDA's curves (footer says QDA); the CV picture -> the version with the
    training set / test set legend; the every-model chart -> one without the random forest, legend clear of the points;
  - compatibility slide: "With in-date ratings: 0.78-0.85" -> 0.78-0.84, and its bar shortened to match;
  - the every-model slide's notes: "to the random forest" -> "to regularized QDA", "0.78 and 0.85" -> "0.78 and 0.84"
    (the only edit to a teammate's notes: the slide no longer shows the forest);
  - outline: "trees" removed from section 3 (the Tree Methods slide is skipped);
  - BIC slide footer cites ISL Sections 6.1.2-6.1.3 (it cited 4.3-4.4, the logistic regression sections);
  - the validation slide's "Our folds are whole events: slide N" points at the CV slide's page;
  - two long footers on Abraham's slides shortened to one line; every shown slide's footer page renumbered;
  - skipped "Methods beyond" slide: "2775 numbers" -> "about 8,000 numbers" (it appears in a PDF export);
  - Abraham's speaker notes only: a definition of a spline on "Logistic regression on our data", and the CV slide's
    tuning example no longer names the random forest (its slide is skipped). Teammates' notes are otherwise untouched.
Run from the repository root: python scripts/report_deck_fixes.py --deck "path/to/Speed-dating midterm presentation.pptx"
"""
import argparse, re
from pptx import Presentation
from pptx.util import Inches
from report_impl_slides import FIG, RES, slide_with, replace_in_frame, replace_text, replace_image

FOOTER = re.compile(r"Math 195 · Speed dating · (\d+)(?=\s|$)")


def find_frame(slide, start):
    hits = [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.startswith(start)]
    assert len(hits) == 1, f"{start!r}: {len(hits)} matches"
    return hits[0].text_frame


def note_replace(slide, old, new):
    tf = slide.notes_slide.notes_text_frame
    assert replace_in_frame(tf, old, new) == 1, f"notes: {old[:40]!r} not found"


def title_of(slide):
    t = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    return next((x for x in t if not x.startswith("Math 195") and not re.match(r"\d ·|[A-Z ]+$", x)), t[0] if t else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--deck", required=True); args = ap.parse_args()
    prs = Presentation(args.deck); slides = list(prs.slides)
    shown = [s for s in slides if s._element.get("show") != "0"]
    page = {s.slide_id: i for i, s in enumerate(shown, 1)}

    main_result = slide_with(prs, "THE MAIN RESULT")
    for old, new in (("0.54", "0.58"), ("0.85", "0.84")):
        big = [sh for sh in main_result.shapes if sh.has_text_frame and sh.text_frame.text.strip() == old]
        assert len(big) == 1; replace_in_frame(big[0].text_frame, old, new)
    replace_text(main_result, "random forest: 0.583 (SE 0.020) before, 0.846 after",
                 "best: QDA 0.579 (SE 0.023) before, lasso 0.838 (SE 0.008) after")
    replace_text(slide_with(prs, "What we will cover"), "models, validation, trees, evaluation", "models, validation, evaluation")
    bic = slide_with(prs, "BIC and Stepwise Logistic Regression")
    replace_text(bic, "An Introduction to Statistical Learning, Sections 4.3–4.4", "An Introduction to Statistical Learning, Sections 6.1.2–6.1.3")
    replace_text(slide_with(prs, "Methods we used beyond"), "(2775 numbers)", "(about 8,000 numbers)")

    cv = slide_with(prs, "Hold out whole events")
    validation = slide_with(prs, "k-Fold Cross-Validation")
    m = re.search(r"Our folds are whole events: slide (\d+)", find_frame(validation, "Math 195").text)
    replace_text(validation, m.group(0), f"Our folds are whole events: slide {page[cv.slide_id]}")
    roc = slide_with(prs, "Model evaluation: the ROC curve")
    replace_image(roc, FIG / "slide_roc_qda.png"); replace_text(roc, "Random forest on held-out events", "Regularized QDA on held-out events")
    replace_image(cv, FIG / "slide_cv_design.png")
    every = slide_with(prs, "Every model is close to chance")
    replace_image(every, FIG / "slide_auc_part1_talk.png")
    note_replace(every, "from logistic regression to the random forest,", "from logistic regression to regularized QDA,")
    note_replace(every, "between 0.78 and 0.85", "between 0.78 and 0.84")

    ladder = slide_with(prs, "Compatibility features add almost nothing")   # its full-scale axis: 0.50 at x0, 1.00 at x1
    replace_text(ladder, "With in-date ratings: 0.78–0.85", "With in-date ratings: 0.78–0.84")
    x0, x1 = (next(sh for sh in ladder.shapes if sh.has_text_frame and sh.text_frame.text == t).left + Inches(0.3) for t in ("0.50", "1.00"))
    bar = [sh for sh in ladder.shapes if sh.shape_type == 1 and abs(sh.left - (x0 + 0.28 / 0.5 * (x1 - x0))) < Inches(0.01)
           and abs(sh.left + sh.width - (x0 + 0.35 / 0.5 * (x1 - x0))) < Inches(0.01)]
    assert len(bar) == 1, f"date-ratings bar: {len(bar)} matches"; bar[0].width = int(0.06 / 0.5 * (x1 - x0))

    logit = slide_with(prs, "Logistic regression on our data")
    tf = find_frame(logit, "Math 195"); old = tf.text.split(" · ", 3)[3]
    replace_in_frame(tf, old, "BIC: ISL §6.1.3 · Splines: ISL §7.4 · Dots: the five pre-date training folds; none chosen with the date ratings either")
    lda = slide_with(prs, "LDA and QDA on our data")
    tf = find_frame(lda, "Math 195"); old = tf.text.split(" · ", 3)[3]
    replace_in_frame(tf, old, "LDA and QDA: ISL §4.4 · Our training folds: 124 to 128 columns, 3,047 to 3,750 pairs")

    note_replace(logit, "and let BIC decide whether it stays.",
                 "and let BIC decide whether it stays. A spline lets a predictor's effect bend instead of following a straight "
                 "line, for example match chances rising with age and then falling. Ours is a natural cubic spline: smooth "
                 "cubic pieces joined at the one-third and two-thirds points of the data, and straight beyond its ends.")
    note_replace(cv, "like the random forest's leaf size,", "like QDA's shrinkage,")

    for s in shown:
        for sh in s.shapes:
            mm = sh.has_text_frame and FOOTER.match(sh.text_frame.text)
            if mm and int(mm.group(1)) != page[s.slide_id]:
                replace_in_frame(sh.text_frame, mm.group(0), f"Math 195 · Speed dating · {page[s.slide_id]}")

    out = RES / "report_deck_fixed.pptx"; prs.save(out); print(f"wrote {out} ({len(shown)} shown, {len(slides) - len(shown)} skipped)")
    for s in slides:
        foot = next((FOOTER.match(sh.text_frame.text).group(1) for sh in s.shapes if sh.has_text_frame and FOOTER.match(sh.text_frame.text)), "-")
        print(f"{page.get(s.slide_id, 'skip'):>4}  footer {foot:>4}  {title_of(s)[:60]}")


if __name__ == "__main__":
    main()
