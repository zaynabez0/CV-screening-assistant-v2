from django.db import models


class JobDescription(models.Model):
    """The JD and its confirmed requirements list.

    NOTE: models only for now. JD extraction and the requirement review/edit UI are a
    later phase — this exists so ScreeningRun (and therefore Candidate) has something
    to point at.
    """

    INPUT_MODE_CHOICES = [("upload", "Upload"), ("text", "Free text")]

    input_mode = models.CharField(max_length=10, choices=INPUT_MODE_CHOICES)
    raw_file = models.FileField(upload_to="jds/", blank=True, null=True)
    raw_text = models.TextField(blank=True)
    # Confirmed flat list; each item: {requirement, description, type, scale, suggested_weight}
    # Defaults to empty: a run starts at jd_pending, before any requirement exists.
    requirements = models.JSONField(default=list)
    confirmed_by_user = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"JD #{self.pk} ({self.get_input_mode_display()})"
