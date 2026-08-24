from django.urls import path

from . import views

app_name = "runs"

urlpatterns = [
    path("run/<int:run_id>/stage1/start/", views.start_stage1, name="stage1_start"),
    path("run/<int:run_id>/stage1/status/", views.stage1_status, name="stage1_status"),
    path("run/<int:run_id>/stage1/review/", views.stage1_review, name="stage1_review"),
    path("run/<int:run_id>/stage1/confirm/", views.confirm_stage1, name="stage1_confirm"),
]
