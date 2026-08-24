from django.contrib import admin

from .models import JobDescription


@admin.register(JobDescription)
class JobDescriptionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "input_mode", "confirmed_by_user", "created_at")
