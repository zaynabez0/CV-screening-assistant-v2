from django.urls import path

from . import views

app_name = "jobs"

urlpatterns = [
    path("runs/new/", views.run_create, name="run_create"),
    path("run/<int:run_id>/jd/review/", views.jd_review, name="jd_review"),
    path("run/<int:run_id>/jd/extract/", views.jd_extract, name="jd_extract"),
    path("run/<int:run_id>/jd/save/", views.jd_save, name="jd_save"),
    path("run/<int:run_id>/jd/add/", views.jd_add, name="jd_add"),
    path("run/<int:run_id>/jd/confirm/", views.jd_confirm, name="jd_confirm"),
]
