from django.contrib import admin

from .models import JobDescription


@admin.register(JobDescription)
class JobDescriptionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "input_mode", "requirement_count", "confirmed_by_user", "created_at")
    list_filter = ("input_mode", "confirmed_by_user")
    readonly_fields = ("created_at",)

    @admin.display(description="Requirements")
    def requirement_count(self, obj):
        return len(obj.requirements or [])
