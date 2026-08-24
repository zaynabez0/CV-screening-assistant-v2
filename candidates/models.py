from django.db import models


class Candidate(models.Model):
    """One uploaded CV and everything the pipeline learns about it.

    `extracted_json` is the CV_SCHEMA output. Once it is set, every downstream stage
    reads it (anonymized) and never touches the raw file again.
    """

    STATUS_CHOICES = [
        ("pending", "Pending validation"),
        ("ready", "Structured, ready for Stage 1"),
        ("rejected", "Rejected"),
        ("duplicate", "Duplicate"),
    ]

    file = models.FileField(upload_to="cvs/")
    original_filename = models.CharField(max_length=255)
    file_hash = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    rejection_reason = models.TextField(blank=True)
    extracted_json = models.JSONField(null=True, blank=True)  # CV_SCHEMA output, once structured
    uploaded_at = models.DateTimeField(auto_now_add=True)
    run = models.ForeignKey("runs.ScreeningRun", on_delete=models.CASCADE)

    class Meta:
        ordering = ["uploaded_at", "pk"]

    def __str__(self):
        return f"{self.original_filename} ({self.get_status_display()})"

    @property
    def extraction_confidence(self) -> str:
        """"high"/"medium"/"low" once structured, "" before. Used by the review checkpoint."""
        if not self.extracted_json:
            return ""
        return self.extracted_json.get("extraction_confidence", "")

    @property
    def is_low_confidence(self) -> bool:
        """Drives the visual flag on the review page — the garbled-extraction catch."""
        return self.extraction_confidence == "low"
