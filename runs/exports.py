"""Excel summary of a screening run — one row per candidate, readable by a hiring manager.

Three sheets:
  "Candidates"  — every candidate, which stages they passed or failed, why they stopped
                  where they did, and the final score for anyone who reached Stage 3.
  "Run summary" — the job description's requirements and the settings the stages ran
                  with, so the numbers on sheet 1 can be interpreted months later.
  "LLM cost"    — what the run cost and how long each step took, per step and per CV.

Reads `candidate_history()`, the same assembled view the Final Review page renders, so
the spreadsheet and the screen can never disagree.

This is the *readable* export. The CSV alongside it carries the full per-requirement
rationale for audit; this one is the summary you send to someone.
"""

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from screening.stage2_ranking.ranking import shortlist_size

from .models import LlmCall, StageResult
from .services import candidate_history
from .usage import STEP_LABELS, candidate_cost_rows, format_cost, format_seconds, run_cost_summary

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
TITLE_FONT = Font(bold=True, size=13)
PASS_FILL = PatternFill("solid", fgColor="E7F4E8")
FAIL_FILL = PatternFill("solid", fgColor="FDECEA")
MUTED_FILL = PatternFill("solid", fgColor="F4F4F4")
FLAG_FILL = PatternFill("solid", fgColor="FFF4E5")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

COLUMNS = [
    ("#", 5),
    ("CV file", 26),
    ("Candidate", 22),
    ("Outcome", 30),
    ("Preprocessing", 15),
    ("Stage 1 — hard filter", 20),
    ("Why they stopped", 46),
    ("Stage 2 rank", 12),
    ("Stage 2 — ranking", 20),
    ("Stage 3 score", 13),
    ("Low-confidence criteria", 30),
    ("Recruiter overrides", 40),
]


def _stage1_cells(row):
    """(verdict, brief reason). The reason names the unmet must-haves, not the essay."""
    result = row["stage1"]
    if result is None:
        return "not reached", ""
    if result.passed:
        return "passed", ""
    unmet = result.unmet_requirements
    if not unmet:
        return "failed", "did not meet the must-have requirements"
    reason = "; ".join(
        f"{e['requirement']} — {e['reason']}" for e in unmet[:3]
    )
    if len(unmet) > 3:
        reason += f" (+{len(unmet) - 3} more)"
    return "failed", reason


def _stage2_cells(row):
    if row["stage2"] is None:
        return "", "not reached"
    return row["stage2_rank"], "shortlisted" if row["shortlisted"] else "below cutoff"


def _preprocessing_cell(candidate):
    if candidate.status == "ready":
        return "ok"
    return candidate.get_status_display().lower()


def _overrides_cell(row):
    return "\n".join(
        f"{o.get_action_display()}"
        + (f" → {o.adjusted_score:g}" if o.adjusted_score is not None else "")
        + f" ({o.performed_by.username}, {o.created_at:%Y-%m-%d}): {o.reason}"
        for o in row["overrides"]
    )


def _write_candidates_sheet(sheet, run, rows):
    sheet.title = "Candidates"

    sheet["A1"] = f"Screening run #{run.pk} — candidate outcomes"
    sheet["A1"].font = TITLE_FONT
    sheet["A2"] = (
        f"{len(rows)} candidate(s). Nobody was auto-rejected: every candidate below is "
        "shown with the reason they stopped where they did."
    )
    sheet["A2"].font = Font(italic=True, color="666666", size=10)

    header_row = 4
    for index, (title, width) in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=header_row, column=index, value=title)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.row_dimensions[header_row].height = 30

    for offset, row in enumerate(rows):
        excel_row = header_row + 1 + offset
        candidate = row["candidate"]
        extracted = candidate.extracted_json or {}
        stage1_verdict, stage1_reason = _stage1_cells(row)
        stage2_rank, stage2_verdict = _stage2_cells(row)

        values = [
            row["position"],
            candidate.original_filename,
            extracted.get("full_name", ""),
            row["outcome"],
            _preprocessing_cell(candidate),
            stage1_verdict,
            stage1_reason or candidate.rejection_reason,
            stage2_rank,
            stage2_verdict,
            row["effective_score"] if row["effective_score"] is not None else "",
            ", ".join(row["low_confidence_flags"]),
            _overrides_cell(row),
        ]
        for index, value in enumerate(values, start=1):
            cell = sheet.cell(row=excel_row, column=index, value=value)
            cell.border = BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=index in (4, 7, 11, 12))

        # Colour the two columns a reader scans first.
        stage1_cell = sheet.cell(row=excel_row, column=6)
        if stage1_verdict == "passed":
            stage1_cell.fill = PASS_FILL
        elif stage1_verdict == "failed":
            stage1_cell.fill = FAIL_FILL
        else:
            stage1_cell.fill = MUTED_FILL

        score_cell = sheet.cell(row=excel_row, column=10)
        score_cell.number_format = "0.0"
        if row["effective_score"] is not None:
            score_cell.fill = PASS_FILL
            score_cell.font = Font(bold=True)
        if row["low_confidence_flags"]:
            sheet.cell(row=excel_row, column=11).fill = FLAG_FILL

    last_row = header_row + len(rows)
    if rows:
        sheet.auto_filter.ref = f"A{header_row}:L{last_row}"
    sheet.freeze_panes = f"A{header_row + 1}"


def _write_summary_sheet(sheet, run, rows):
    sheet.title = "Run summary"
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 62
    sheet.column_dimensions["C"].width = 16
    sheet.column_dimensions["D"].width = 46

    sheet["A1"] = f"Screening run #{run.pk}"
    sheet["A1"].font = TITLE_FONT

    scored = [r for r in rows if r["stage3"] is not None]
    ranked = [r for r in rows if r["stage2"] is not None]
    cost = run_cost_summary(run)
    facts = [
        ("Current stage", run.get_current_stage_display()),
        ("Created", run.created_at.strftime("%Y-%m-%d %H:%M")),
        ("Candidates uploaded", len(rows)),
        ("Passed Stage 1", sum(1 for r in rows if r["stage1"] is not None and r["stage1"].passed)),
        ("Filtered out at Stage 1", sum(1 for r in rows if r["stage1"] is not None and not r["stage1"].passed)),
        ("Ranked at Stage 2", len(ranked)),
        ("Shortlist cutoff", f"{run.stage2_cutoff_percent:g}%  →  top {shortlist_size(len(ranked), run.stage2_cutoff_percent)}"),
        ("Stage 2 keywords", ", ".join(run.stage2_keywords) or "(none)"),
        ("Scored at Stage 3", len(scored)),
        ("Recruiter overrides", sum(len(r["overrides"]) for r in rows)),
        ("LLM calls made", cost["total"]["calls"]),
        ("Total inference time", cost["total"]["seconds_display"]),
        ("Total LLM cost", cost["total"]["cost_display"]),
    ]
    for offset, (label, value) in enumerate(facts):
        row = 3 + offset
        sheet.cell(row=row, column=1, value=label).font = Font(bold=True)
        sheet.cell(row=row, column=2, value=value)

    start = 3 + len(facts) + 2
    sheet.cell(row=start - 1, column=1, value="Job description requirements").font = TITLE_FONT
    for index, title in enumerate(["Requirement", "Description", "Type", "Scale (Stage 3)"], start=1):
        cell = sheet.cell(row=start, column=index, value=title)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT

    for offset, requirement in enumerate(run.job_description.requirements or [], start=1):
        row = start + offset
        for index, key in enumerate(["requirement", "description", "type", "scale"], start=1):
            cell = sheet.cell(row=row, column=index, value=requirement.get(key, ""))
            cell.border = BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=index in (2, 4))
        if requirement.get("type") == "must_have":
            sheet.cell(row=row, column=3).fill = FAIL_FILL   # gated: the strictest type


COST_STEPS = LlmCall.STEP_ORDER


def _write_cost_sheet(sheet, run, rows):
    """What the run cost, per step and then per candidate.

    The per-step block answers "where did the money and the time go"; the per-candidate
    block below it is the inference time for one CV at each step, which the on-screen
    panel only shows as an average.
    """
    sheet.title = "LLM cost"
    sheet.column_dimensions["A"].width = 30
    for letter in "BCDEF":
        sheet.column_dimensions[letter].width = 15
    sheet.column_dimensions["B"].width = 26          # candidate names in the lower block

    sheet["A1"] = f"Screening run #{run.pk} — LLM cost and inference time"
    sheet["A1"].font = TITLE_FONT

    cost = run_cost_summary(run)
    if not cost["has_calls"]:
        sheet["A3"] = "No LLM calls have been recorded for this run yet."
        return

    if cost["unpriced_models"]:
        sheet["A2"] = ("No price is configured for "
                       + ", ".join(cost["unpriced_models"])
                       + " — token counts and timings are exact, cost is not available.")
        sheet["A2"].font = Font(italic=True, color="808080")

    header_row = 4
    for index, title in enumerate(["Step", "Calls", "Tokens", "Time", "Per CV", "Cost"], start=1):
        cell = sheet.cell(row=header_row, column=index, value=title)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT

    for offset, row in enumerate(cost["rows"], start=1):
        values = [row["label"], row["calls"], row["tokens"],
                  row["seconds_display"], row["per_cv_display"], row["cost_display"]]
        for index, value in enumerate(values, start=1):
            sheet.cell(row=header_row + offset, column=index, value=value).border = BORDER

    total_row = header_row + len(cost["rows"]) + 1
    totals = ["Total", cost["total"]["calls"], cost["total"]["tokens"],
              cost["total"]["seconds_display"], "—", cost["total"]["cost_display"]]
    for index, value in enumerate(totals, start=1):
        cell = sheet.cell(row=total_row, column=index, value=value)
        cell.font, cell.fill, cell.border = Font(bold=True), MUTED_FILL, BORDER

    # --- per candidate, per step -------------------------------------------------
    start = total_row + 3
    sheet.cell(row=start - 1, column=1, value="Per candidate").font = TITLE_FONT
    per_candidate = candidate_cost_rows(run)
    steps = [s for s in COST_STEPS if any(s in v for v in per_candidate.values())]

    titles = ["CV file"] + [f"{STEP_LABELS[s]} — time" for s in steps] + \
             [f"{STEP_LABELS[s]} — cost" for s in steps]
    for index, title in enumerate(titles, start=1):
        cell = sheet.cell(row=start, column=index, value=title)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        sheet.column_dimensions[get_column_letter(index)].width = max(16, len(title) + 2)

    for offset, row in enumerate(rows, start=1):
        by_step = per_candidate.get(row["candidate"].pk, {})
        values = [row["candidate"].original_filename]
        values += [format_seconds(by_step[s]["seconds"]) if s in by_step else "" for s in steps]
        values += [format_cost(by_step[s]["cost"]) if s in by_step else "" for s in steps]
        for index, value in enumerate(values, start=1):
            sheet.cell(row=start + offset, column=index, value=value).border = BORDER

    sheet.freeze_panes = "B1"


def build_summary_workbook(run) -> Workbook:
    """The whole run as a three-sheet workbook."""
    rows = candidate_history(run)

    workbook = Workbook()
    _write_candidates_sheet(workbook.active, run, rows)
    _write_summary_sheet(workbook.create_sheet(), run, rows)
    _write_cost_sheet(workbook.create_sheet(), run, rows)
    return workbook
