"""JD Processing: text or file in, a reviewed requirements list out.

This is the first step of the pipeline and the application's entry point. Nothing
downstream reads an unconfirmed requirements list, and `jd_confirm` is the only view
here that touches `current_stage` — extraction never advances the run on its own.
"""

from django.contrib import messages
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render

from jobs.models import JobDescription
from runs.models import ScreeningRun
from runs.usage import record_step, step_cost
from screening.cv.validation import CVValidationError, check_extension, check_magic_type
from screening.jd.extraction import extract_requirements, get_jd_text, normalize_requirement
from screening.jd.schema import REQUIREMENT_TYPES

REQUIRED_STAGE = "jd_pending"


def _blocked_reason(run):
    """JD editing is available only while the run is at jd_pending and unconfirmed."""
    if run.current_stage != REQUIRED_STAGE:
        return (
            f"This run is at '{run.get_current_stage_display()}'. The job description was "
            f"confirmed and is now locked — editing it would leave the stages that already "
            f"ran judged against requirements that no longer exist."
        )
    return None


def run_create(request):
    """GET: the JD input form. POST: create the JobDescription and its ScreeningRun.

    The two are created together in a transaction: ScreeningRun.job_description is a
    non-null PROTECT foreign key, so a run cannot exist without one.
    """
    if request.method != "POST":
        return render(request, "jobs/run_new.html", {})

    mode = request.POST.get("input_mode", "text")
    free_text = (request.POST.get("raw_text") or "").strip()
    uploaded = request.FILES.get("raw_file")

    if mode == "upload":
        if not uploaded:
            messages.error(request, "Choose a PDF or DOCX file, or switch to free text.")
            return render(request, "jobs/run_new.html", {"raw_text": free_text})
        try:
            check_extension(uploaded.name)
        except CVValidationError as exc:
            messages.error(request, f"Job description rejected: {exc.reason}")
            return render(request, "jobs/run_new.html", {"raw_text": free_text})
    elif not free_text:
        messages.error(request, "Paste the job description text, or switch to file upload.")
        return render(request, "jobs/run_new.html", {})

    with transaction.atomic():
        jd = JobDescription.objects.create(
            input_mode="upload" if mode == "upload" else "text",
            raw_file=uploaded if mode == "upload" else None,
            raw_text="" if mode == "upload" else free_text,
            requirements=[],
            confirmed_by_user=False,
        )
        run = ScreeningRun.objects.create(job_description=jd)  # defaults to jd_pending

    # Validate the saved file's real content, and fail the whole creation if unreadable.
    if mode == "upload":
        try:
            check_magic_type(jd.raw_file.path, check_extension(uploaded.name))
            text = get_jd_text(uploaded_file=jd.raw_file)
        except (CVValidationError, ValueError, RuntimeError) as exc:
            reason = getattr(exc, "reason", f"{type(exc).__name__}: {exc}")
            run.delete()
            jd.delete()
            messages.error(request, f"Job description rejected: {reason}")
            return render(request, "jobs/run_new.html", {})

        from screening.common.text_extraction import has_extractable_text
        if not has_extractable_text(text):
            run.delete()
            jd.delete()
            messages.error(
                request,
                "Job description rejected: no extractable text found. A scanned PDF has no "
                "text layer, and there is no OCR in v1.",
            )
            return render(request, "jobs/run_new.html", {})

    messages.success(request, f"Run #{run.pk} created. Extract the requirements next.")
    return redirect("jobs:jd_review", run_id=run.pk)


def jd_review(request, run_id):
    """The JD review checkpoint: extract, then edit the list before confirming."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    jd = run.job_description
    requirements = list(jd.requirements or [])

    return render(request, "jobs/jd_review.html", {
        "run": run,
        "jd": jd,
        "requirements": [dict(r, index=i) for i, r in enumerate(requirements)],
        "requirement_types": REQUIREMENT_TYPES,
        "blocked": _blocked_reason(run),
        "locked": jd.confirmed_by_user,
        "step_cost": step_cost(run, "jd_extraction"),
        "must_have_count": sum(1 for r in requirements if r.get("type") == "must_have"),
        "has_requirements": bool(requirements),
    })


def jd_extract(request, run_id):
    """POST-only. Sends the JD to the LLM and replaces the requirements list.

    Runs inline: this is a single call, unlike Stages 1 and 3 which make one per
    candidate. Re-running discards any edits, which the page warns about.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("jobs:jd_review", run_id=run.pk)

    blocked = _blocked_reason(run)
    if blocked:
        messages.error(request, blocked)
        return redirect("jobs:jd_review", run_id=run.pk)

    jd = run.job_description
    try:
        if jd.input_mode == "upload":
            jd_text = get_jd_text(uploaded_file=jd.raw_file)
        else:
            jd_text = get_jd_text(free_text=jd.raw_text)
        with record_step(run, "jd_extraction"):
            extracted = extract_requirements(jd_text)
        requirements = [normalize_requirement(r) for r in extracted]
    except Exception as exc:  # noqa: BLE001 — an API failure is not the user's fault
        messages.error(request, f"Extraction failed: {type(exc).__name__}: {exc}")
        return redirect("jobs:jd_review", run_id=run.pk)

    jd.requirements = requirements
    jd.save(update_fields=["requirements"])
    messages.success(
        request,
        f"Extracted {len(requirements)} requirement(s). Review the types before confirming — "
        "this categorisation drives both Stage 1 and Stage 3.",
    )
    return redirect("jobs:jd_review", run_id=run.pk)


def jd_save(request, run_id):
    """POST-only. Rewrites the whole list from the edit form: edits and deletions."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("jobs:jd_review", run_id=run.pk)

    blocked = _blocked_reason(run)
    if blocked:
        messages.error(request, blocked)
        return redirect("jobs:jd_review", run_id=run.pk)

    jd = run.job_description
    kept, deleted = [], 0
    for index in range(len(jd.requirements or [])):
        if request.POST.get(f"req-{index}-delete"):
            deleted += 1
            continue

        name = (request.POST.get(f"req-{index}-requirement") or "").strip()
        if not name:
            messages.error(request, "A requirement cannot be blank. Delete it instead.")
            return redirect("jobs:jd_review", run_id=run.pk)

        req_type = request.POST.get(f"req-{index}-type", "")
        if req_type not in REQUIREMENT_TYPES:
            messages.error(request, f"'{req_type}' is not a valid requirement type.")
            return redirect("jobs:jd_review", run_id=run.pk)

        raw_weight = (request.POST.get(f"req-{index}-suggested_weight") or "0").strip()
        try:
            weight = float(raw_weight or 0)
        except ValueError:
            messages.error(request, f"'{raw_weight}' is not a valid weight for “{name}”.")
            return redirect("jobs:jd_review", run_id=run.pk)

        kept.append(normalize_requirement({
            "requirement": name,
            "description": (request.POST.get(f"req-{index}-description") or "").strip(),
            "type": req_type,
            "scale": (request.POST.get(f"req-{index}-scale") or "").strip(),
            "suggested_weight": weight,
        }))

    jd.requirements = kept
    jd.save(update_fields=["requirements"])
    note = f", {deleted} deleted" if deleted else ""
    messages.success(request, f"Saved {len(kept)} requirement(s){note}.")
    return redirect("jobs:jd_review", run_id=run.pk)


def jd_add(request, run_id):
    """POST-only. Appends one requirement the recruiter wrote themselves."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("jobs:jd_review", run_id=run.pk)

    blocked = _blocked_reason(run)
    if blocked:
        messages.error(request, blocked)
        return redirect("jobs:jd_review", run_id=run.pk)

    name = (request.POST.get("new-requirement") or "").strip()
    if not name:
        messages.error(request, "Give the new requirement a name.")
        return redirect("jobs:jd_review", run_id=run.pk)

    req_type = request.POST.get("new-type", "")
    if req_type not in REQUIREMENT_TYPES:
        messages.error(request, f"'{req_type}' is not a valid requirement type.")
        return redirect("jobs:jd_review", run_id=run.pk)

    raw_weight = (request.POST.get("new-suggested_weight") or "0").strip()
    try:
        weight = float(raw_weight or 0)
    except ValueError:
        messages.error(request, f"'{raw_weight}' is not a valid weight.")
        return redirect("jobs:jd_review", run_id=run.pk)

    jd = run.job_description
    jd.requirements = list(jd.requirements or []) + [normalize_requirement({
        "requirement": name,
        "description": (request.POST.get("new-description") or "").strip(),
        "type": req_type,
        "scale": (request.POST.get("new-scale") or "").strip(),
        "suggested_weight": weight,
    })]
    jd.save(update_fields=["requirements"])
    messages.success(request, f"Added “{name}”.")
    return redirect("jobs:jd_review", run_id=run.pk)


def jd_confirm(request, run_id):
    """POST-only. Locks the requirements list and opens CV upload.

    The only view in this app that writes current_stage = cv_pending.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("jobs:jd_review", run_id=run.pk)

    blocked = _blocked_reason(run)
    if blocked:
        messages.error(request, blocked)
        return redirect("jobs:jd_review", run_id=run.pk)

    jd = run.job_description
    requirements = list(jd.requirements or [])
    if not requirements:
        messages.error(
            request,
            "Extract or add at least one requirement before confirming — an empty list "
            "would pass every candidate through Stage 1 and skip Stage 3 entirely.",
        )
        return redirect("jobs:jd_review", run_id=run.pk)

    with transaction.atomic():
        jd.confirmed_by_user = True
        jd.save(update_fields=["confirmed_by_user"])
        run.current_stage = "cv_pending"
        run.save(update_fields=["current_stage"])

    if not any(r.get("type") == "must_have" for r in requirements):
        messages.warning(
            request,
            "No must-have requirements: Stage 1 has no hard gate, so every candidate "
            "will pass it. That is allowed, but make sure it is what you intended.",
        )
    messages.success(
        request,
        f"Job description confirmed with {len(requirements)} requirement(s). Upload CVs next.",
    )
    return redirect("candidates:upload", run_id=run.pk)
