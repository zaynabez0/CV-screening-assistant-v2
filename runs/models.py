from django.db import models


class ScreeningRun(models.Model):
    """One screening exercise: one JD, a batch of CVs, and the stages they pass through.

    `current_stage` is the single source of truth for what the user can do right now.
    No stage ever advances it automatically — the user confirms each review checkpoint.

    NOTE: the stage-trigger views, StageResult and Override are a later phase. This
    model exists now so Candidate has a run to belong to.
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

    current_stage = models.CharField(max_length=20, choices=STAGE_CHOICES, default="jd_pending")
    job_description = models.ForeignKey("jobs.JobDescription", on_delete=models.PROTECT)
    stage2_keywords = models.JSONField(default=list)
    stage2_cutoff_percent = models.FloatField(default=25.0)
    rubric_weights = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Run #{self.pk} ({self.get_current_stage_display()})"
