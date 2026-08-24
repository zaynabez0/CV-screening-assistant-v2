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
    path("run/<int:run_id>/stage3/review/", views.stage3_review, name="stage3_review"),
    path("run/<int:run_id>/stage3/weights/", views.save_stage3_weights, name="stage3_weights"),
    path("run/<int:run_id>/stage3/weights/reset/", views.reset_stage3_weights, name="stage3_weights_reset"),
    path("run/<int:run_id>/stage3/start/", views.start_stage3, name="stage3_start"),
    path("run/<int:run_id>/stage3/skip/", views.skip_stage3, name="stage3_skip"),
    path("run/<int:run_id>/stage3/confirm/", views.confirm_stage3, name="stage3_confirm"),
    path("run/<int:run_id>/review/", views.final_review, name="final_review"),
    path("run/<int:run_id>/review/override/", views.create_override, name="create_override"),
    path("run/<int:run_id>/review/export.csv", views.export_csv, name="export_csv"),
]
