from django.contrib import admin

from .models import Candidate


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "run", "status", "extraction_confidence", "uploaded_at")
    list_filter = ("status", "run")
    search_fields = ("original_filename", "file_hash")
    readonly_fields = ("file_hash", "uploaded_at")
