from django.urls import path
from . import views
from django.http import HttpResponse

urlpatterns = [
	path('', views.home, name='home'),
	path('predict/', views.predict, name='predict'),
	path('predict/result/', views.result, name='result'),
]