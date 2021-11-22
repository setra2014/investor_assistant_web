from django.urls import path
from django.contrib.auth.views import LogoutView 
from django.contrib.auth.views import LoginView 

from . import views

urlpatterns = [
    path("create_portfolio/", views.create_portfolio, name="create_portfolio"),
    path("", views.get_portfolios, name="get_portfolios"),
]