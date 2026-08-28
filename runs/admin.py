from django.contrib import admin

from .models import LlmCall, ModelPrice, Override, ScreeningRun, StageResult


@admin.register(ScreeningRun)
class ScreeningRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "current_stage", "job_status", "job_stage", "job_processed", "job_total", "created_at")
    list_filter = ("current_stage", "job_status")


@admin.register(StageResult)
class StageResultAdmin(admin.ModelAdmin):
    list_display = ("candidate", "run", "stage", "passed", "created_at")
    list_filter = ("stage", "passed", "run")
    readonly_fields = ("created_at",)


@admin.register(Override)
class OverrideAdmin(admin.ModelAdmin):
    list_display = ("candidate", "run", "action", "performed_by", "created_at")
    list_filter = ("action", "run")
    readonly_fields = ("created_at",)


@admin.register(ModelPrice)
class ModelPriceAdmin(admin.ModelAdmin):
    """Where the operator enters what their key is actually billed.

    Editable straight from the list so all three rates for a model can be corrected in
    one go — the numbers are the whole reason this model exists, and a change here
    re-prices every past run, since cost is computed from stored token counts.
    """

    list_display = ("model_name", "input_per_million", "cached_input_per_million",
                    "output_per_million", "updated_at")
    list_editable = ("input_per_million", "cached_input_per_million", "output_per_million")
    readonly_fields = ("updated_at",)


@admin.register(LlmCall)
class LlmCallAdmin(admin.ModelAdmin):
    list_display = ("run", "step", "candidate", "model_name", "input_tokens",
                    "cached_input_tokens", "output_tokens", "duration_ms", "created_at")
    list_filter = ("step", "model_name", "run")
    readonly_fields = ("created_at",)
