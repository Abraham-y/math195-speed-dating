"""Fixes for the team's midterm deck once it already holds Abraham's implementation slides (the Google Slides version
exported as .pptx), then a page-by-page check of the slides that will be shown.

Input : the team deck as .pptx (--deck); report/figures/slide_roc.png, slide_cv_design.png, slide_auc_part1.png
        (scripts/report_figures.py --slides)
Output: results/report_deck_fixed.pptx, and the check printed (page, title, footer page, pictures replaced).
Fixes, each asserted to apply exactly where expected:
  - main result 0.54 -> 0.58 (best pre-date AUC, random forest 0.583; results/part1_models_metrics.csv);
  - three images: the stock ROC curve -> our random forest's curves; the CV picture -> the version with the
    training set / test set legend; the every-model chart -> the uncropped one (random forest row included,
    legend clear of the points);
  - outline: "trees" removed from section 3 (the Tree Methods slide is skipped);
  - BIC slide footer cites ISL Sections 6.1.2-6.1.3 (it cited 4.3-4.4, the logistic regression sections);
  - the validation slide's "Our folds are whole events: slide N" points at the CV slide's page;
  - two long footers on Abraham's slides shortened to one line; every shown slide's footer page renumbered;
  - skipped "Methods beyond" slide: "2775 numbers" -> "about 8,000 numbers" (it appears in a PDF export);
  - Abraham's speaker notes only: a definition of a spline on "Logistic regression on our data", and the CV slide's
    tuning example no longer names the random forest (its slide is skipped). Teammates' notes are not touched.
Run from the repository root: python scripts/report_deck_fixes.py --deck "path/to/Speed-dating midterm presentation.pptx"
"""
import argparse, re
from pptx import Presentation
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
    big = [sh for sh in main_result.shapes if sh.has_text_frame and sh.text_frame.text.strip() == "0.54"]
    assert len(big) == 1; replace_in_frame(big[0].text_frame, "0.54", "0.58")
    replace_text(slide_with(prs, "What we will cover"), "models, validation, trees, evaluation", "models, validation, evaluation")
    bic = slide_with(prs, "BIC and Stepwise Logistic Regression")
    replace_text(bic, "An Introduction to Statistical Learning, Sections 4.3–4.4", "An Introduction to Statistical Learning, Sections 6.1.2–6.1.3")
    replace_text(slide_with(prs, "Methods we used beyond"), "(2775 numbers)", "(about 8,000 numbers)")

    cv = slide_with(prs, "Hold out whole events")
    validation = slide_with(prs, "k-Fold Cross-Validation")
    m = re.search(r"Our folds are whole events: slide (\d+)", find_frame(validation, "Math 195").text)
    replace_text(validation, m.group(0), f"Our folds are whole events: slide {page[cv.slide_id]}")
    replace_image(slide_with(prs, "Model evaluation: the ROC curve"), FIG / "slide_roc.png")
    replace_image(cv, FIG / "slide_cv_design.png")
    replace_image(slide_with(prs, "Every model is close to chance"), FIG / "slide_auc_part1.png")

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
