from django.contrib import admin

from .models import ScreeningRun, StageResult


@admin.register(ScreeningRun)
class ScreeningRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "current_stage", "job_status", "job_stage", "job_processed", "job_total", "created_at")
    list_filter = ("current_stage", "job_status")


@admin.register(StageResult)
class StageResultAdmin(admin.ModelAdmin):
    list_display = ("candidate", "run", "stage", "passed", "created_at")
    list_filter = ("stage", "passed", "run")
    readonly_fields = ("created_at",)
