from decimal import Decimal

from django.db import models


class ScreeningRun(models.Model):
    """One screening exercise: one JD, a batch of CVs, and the stages they pass through.

    `current_stage` is the single source of truth for what the user can do right now.
    No stage ever advances it automatically — the user confirms each review checkpoint.

    The job_* fields are the async-job status the frontend polls. A stage runs in a
    subprocess-launched management command; that process writes progress here, and the
    browser reads it back. No Celery in v1.

    NOTE: Override (the human intervention log) belongs to Final Review and is not
    built yet.
    """

    STAGE_CHOICES = [
        ("jd_pending", "Awaiting JD confirmation"),
        ("cv_pending", "Awaiting CV upload and preprocessing"),
        ("stage1_pending", "Ready to run hard filter"),
        ("stage2_pending", "Ready to run ranking"),
        ("stage3_pending", "Ready to set weights and run scoring"),
        ("review", "Ready for final review"),
        ("done", "Closed"),
    ]

    # The stage keys in pipeline order. Lets a checkpoint ask "has the run moved past me?"
    # without every caller hard-coding the sequence.
    STAGE_ORDER = [key for key, _label in STAGE_CHOICES]

    JOB_STATUS_CHOICES = [
        ("idle", "No job running"),
        ("running", "Running"),
        ("done", "Finished"),
        ("failed", "Failed"),
    ]

    current_stage = models.CharField(max_length=20, choices=STAGE_CHOICES, default="jd_pending")
    job_description = models.ForeignKey("jobs.JobDescription", on_delete=models.PROTECT)
    stage2_keywords = models.JSONField(default=list)
    stage2_cutoff_percent = models.FloatField(default=25.0)
    rubric_weights = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    # --- async job status, polled by the frontend ---
    job_status = models.CharField(max_length=10, choices=JOB_STATUS_CHOICES, default="idle")
    job_stage = models.CharField(max_length=10, blank=True)   # which stage the job is for
    job_message = models.TextField(blank=True)                # human-readable; the error text on failure
    job_processed = models.PositiveIntegerField(default=0)
    job_total = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Run #{self.pk} ({self.get_current_stage_display()})"

    def has_passed(self, stage: str) -> bool:
        """True once the run has moved *beyond* `stage` — that checkpoint is behind it.

        Strictly after, never at: a run sitting on `stage` still has work to do there.
        Review pages use this to swap their action forms for a link onward, so a
        confirmed checkpoint is a finished step rather than a dead end.
        """
        return self.STAGE_ORDER.index(self.current_stage) > self.STAGE_ORDER.index(stage)

    @property
    def job_percent(self) -> int:
        if not self.job_total:
            return 0
        return int(self.job_processed / self.job_total * 100)

    def mark_job_started(self, stage: str, total: int) -> None:
        self.job_status = "running"
        self.job_stage = stage
        self.job_message = ""
        self.job_processed = 0
        self.job_total = total
        self.save(update_fields=["job_status", "job_stage", "job_message", "job_processed", "job_total"])

    def mark_job_progress(self, processed: int) -> None:
        self.job_processed = processed
        self.save(update_fields=["job_processed"])

    def mark_job_finished(self, message: str = "") -> None:
        self.job_status = "done"
        self.job_message = message
        self.save(update_fields=["job_status", "job_message"])

    def mark_job_failed(self, message: str) -> None:
        self.job_status = "failed"
        self.job_message = message
        self.save(update_fields=["job_status", "job_message"])


class StageResult(models.Model):
    """The automated audit trail: one row per candidate per stage they passed through.

    Nobody edits these after the fact — a human change is a separate Override row, so
    the two together answer "what did the pipeline decide, and what did a human change".
    """

    STAGE_CHOICES = [("stage1", "Hard filter"), ("stage2", "Ranking"), ("stage3", "Scoring")]

    candidate = models.ForeignKey("candidates.Candidate", on_delete=models.CASCADE)
    run = models.ForeignKey("runs.ScreeningRun", on_delete=models.CASCADE)
    stage = models.CharField(max_length=10, choices=STAGE_CHOICES)
    passed = models.BooleanField(null=True)   # null where pass/fail doesn't apply (e.g. Stage 2 ranking)
    output_json = models.JSONField()          # stage1: evaluations list; stage2: scores+rank; stage3: criterion_breakdown+final_score
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["candidate_id", "stage"]

    def __str__(self):
        return f"{self.get_stage_display()} — {self.candidate.original_filename}"

    @property
    def evaluations(self) -> list:
        """Stage 1's per-requirement list. Empty for other stages."""
        if self.stage != "stage1":
            return []
        return self.output_json.get("evaluations", [])

    @property
    def unmet_requirements(self) -> list:
        return [e for e in self.evaluations if not e.get("met")]


class Override(models.Model):
    """The human intervention log — sparse, created only on explicit recruiter action.

    Overrides never edit a StageResult. The automated audit trail stays exactly as the
    pipeline wrote it, and the effective outcome is StageResult + Override computed on
    read. That separation is what lets the record answer, per candidate: what did the
    pipeline decide, and what did a human change about it afterwards.

    This is also the GDPR Art. 22 mechanism: no candidate is auto-rejected, because
    every filtered-out candidate stays visible and reversible from Final Review.
    """

    ACTION_CHOICES = [("reinstate", "Reinstate filtered candidate"), ("adjust_score", "Adjust score"), ("note", "Add note")]

    candidate = models.ForeignKey("candidates.Candidate", on_delete=models.CASCADE)
    run = models.ForeignKey("runs.ScreeningRun", on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    reason = models.TextField()
    # Added beyond CLAUDE.md's model: "adjust_score" has no value to apply without it.
    # Null for the other two actions.
    adjusted_score = models.FloatField(null=True, blank=True)
    performed_by = models.ForeignKey("auth.User", on_delete=models.PROTECT)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-pk"]

    def __str__(self):
        return f"{self.get_action_display()} — {self.candidate.original_filename}"


class ModelPrice(models.Model):
    """What one model costs, in EUR per million tokens. Edited in the Django admin.

    Deliberately data rather than code: the rate a gateway-issued sub-key is billed at is
    the operator's to know, and hardcoding a guess for a model this app does not own would
    produce confident, wrong numbers. With no row for a model, cost reads as "not priced"
    rather than as zero.

    Three rates because the Responses API bills three ways, and the middle one is the
    whole point of this project's cached-prefix prompt design: a cached input token is
    typically a fraction of a fresh one.
    """

    model_name = models.CharField(max_length=100, unique=True)  # e.g. "gpt-5.6-luna"
    input_per_million = models.DecimalField(max_digits=12, decimal_places=4)
    cached_input_per_million = models.DecimalField(max_digits=12, decimal_places=4)
    output_per_million = models.DecimalField(max_digits=12, decimal_places=4)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["model_name"]

    def __str__(self):
        return f"{self.model_name} (in {self.input_per_million}, out {self.output_per_million} €/Mtok)"

    def cost_for(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> Decimal:
        """EUR for one call, or for a summed group of them — the formula is linear, so
        totalling tokens first and pricing once gives the same answer as pricing each call.

        `input_tokens` is the total INCLUDING the cached ones, so the cached share is
        subtracted before the full rate is applied. Adding the two instead would overcharge
        every repeated prefix, which is exactly what this app maximises.
        """
        fresh_input = max(0, input_tokens - cached_input_tokens)
        total = (
            fresh_input * self.input_per_million
            + cached_input_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
        )
        return total / Decimal(1_000_000)


class LlmCall(models.Model):
    """One LLM request: what it cost in tokens, and how long it took.

    One row per call rather than a running total on the run, because "inference time per
    CV at each step" needs the individual calls to average, and because the stages run in
    a separate process whose only channel back is the database.

    `candidate` is null for JD extraction, which is one call for the whole run rather than
    one per CV. Cost is NOT stored: it is computed from these token counts against the
    current ModelPrice, so entering a rate after a run has finished prices that run
    retroactively instead of freezing it at zero.
    """

    STEP_CHOICES = [
        ("jd_extraction", "JD extraction"),
        ("cv_structuring", "CV structuring"),
        ("stage1", "Stage 1 — hard filter"),
        ("stage2", "Stage 2 — ranking"),
        ("stage3", "Stage 3 — scoring"),
    ]
    STEP_ORDER = [key for key, _label in STEP_CHOICES]

    run = models.ForeignKey("runs.ScreeningRun", on_delete=models.CASCADE)
    candidate = models.ForeignKey("candidates.Candidate", on_delete=models.CASCADE,
                                  null=True, blank=True)  # null: JD extraction is per-run
    step = models.CharField(max_length=20, choices=STEP_CHOICES)
    model_name = models.CharField(max_length=100)
    input_tokens = models.PositiveIntegerField(default=0)
    cached_input_tokens = models.PositiveIntegerField(default=0)  # a subset of input_tokens
    output_tokens = models.PositiveIntegerField(default=0)
    reasoning_tokens = models.PositiveIntegerField(default=0)     # a subset of output_tokens
    duration_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "pk"]

    def __str__(self):
        return f"{self.get_step_display()} · {self.duration_ms} ms · {self.total_tokens} tokens"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
