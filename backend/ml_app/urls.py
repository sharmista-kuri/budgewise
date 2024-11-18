# ml_app/urls.py
from django.urls import path
from .views import ExpenseForecastView, LoanEligibilityView

urlpatterns = [
    path('predict-expense/', ExpenseForecastView.as_view(), name='predict-expense'),
    path('predict-loan-eligibility/', LoanEligibilityView.as_view(), name='predict-loan-eligibility'),
]
