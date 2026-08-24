from django.urls import path

from . import views

app_name = "runs"

urlpatterns = [
    path("run/<int:run_id>/stage1/start/", views.start_stage1, name="stage1_start"),
    path("run/<int:run_id>/job/status/", views.job_status, name="job_status"),
    path("run/<int:run_id>/stage1/review/", views.stage1_review, name="stage1_review"),
    path("run/<int:run_id>/stage1/confirm/", views.confirm_stage1, name="stage1_confirm"),
    path("run/<int:run_id>/stage2/review/", views.stage2_review, name="stage2_review"),
    path("run/<int:run_id>/stage2/settings/", views.save_stage2_settings, name="stage2_settings"),
    path("run/<int:run_id>/stage2/reseed/", views.reseed_stage2_keywords, name="stage2_reseed"),
    path("run/<int:run_id>/stage2/start/", views.start_stage2, name="stage2_start"),
    path("run/<int:run_id>/stage2/confirm/", views.confirm_stage2, name="stage2_confirm"),
]
