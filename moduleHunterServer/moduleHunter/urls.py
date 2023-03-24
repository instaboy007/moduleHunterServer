from django.urls import path
from . import views

urlpatterns = [
    path('getModulesInfo', views.getModulesInfo, name='getModulesInfo'),
    path('search', views.modulesSearch, name='modulesSearch'),
]