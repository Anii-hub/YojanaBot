"""finder app URL patterns."""

from django.urls import path
from . import views

app_name = "finder"

urlpatterns = [
    path("",           views.home,         name="home"),
    path("find/",      views.find,         name="find"),
    path("results/",   views.results,      name="results"),
    path("about/",     views.about,        name="about"),
    path("health/",    views.health_check, name="health"),
]
