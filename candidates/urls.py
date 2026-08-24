from django.urls import path

from . import views

app_name = "candidates"

urlpatterns = [
    path("run/<int:run_id>/cvs/upload/", views.upload_cvs, name="upload"),
    path("run/<int:run_id>/cvs/structure/", views.structure_cvs, name="structure"),
    path("run/<int:run_id>/cvs/review/", views.review_candidates, name="review"),
    path("run/<int:run_id>/cvs/confirm/", views.confirm_preprocessing, name="confirm"),
]
