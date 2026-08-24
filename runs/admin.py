from django.contrib import admin

from .models import ScreeningRun


@admin.register(ScreeningRun)
class ScreeningRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "current_stage", "job_description", "created_at")
    list_filter = ("current_stage",)
