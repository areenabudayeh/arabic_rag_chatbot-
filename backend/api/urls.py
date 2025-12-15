from django.urls import path
from .views import health_check, query_view, ingest_view

urlpatterns = [
    path("health/", health_check, name="health"),
    path("query/", query_view, name="query"),
    path("ingest/", ingest_view, name="ingest"),
]
