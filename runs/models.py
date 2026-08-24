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
