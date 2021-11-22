from django import forms
from django.forms import ModelForm

from .models import *

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model
User = get_user_model()


class PortfolioForm(ModelForm):
    class Meta:
        model = PortfolioInfo
        fields = ("title", "criteria", "model")
        labels = {
            "title": "Название",
            "criteria": "Критерий",
            "model": "Модель",
        }

    stocks = forms.CharField()
    deposit = forms.IntegerField()
    min_klaster = forms.IntegerField()
    max_klaster = forms.IntegerField()

