# ml_app/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ExpenseForecastSerializer, LoanEligibilitySerializer
from .predict import predict_expenses, predict_loan_eligibility

class ExpenseForecastView(APIView):
    def post(self, request):
        serializer = ExpenseForecastSerializer(data=request.data)
        if serializer.is_valid():
            prediction = predict_expenses(serializer.validated_data['data'])
            return Response({"prediction": prediction})
        return Response(serializer.errors, status=400)

class LoanEligibilityView(APIView):
    def post(self, request):
        serializer = LoanEligibilitySerializer(data=request.data)
        if serializer.is_valid():
            eligibility = predict_loan_eligibility(serializer.validated_data)
            return Response({"eligibility": eligibility})
        return Response(serializer.errors, status=400)
