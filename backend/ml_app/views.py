# ml_app/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ExpenseForecastSerializer, LoanEligibilitySerializer, GroupExpenseForecastSerializer, AnomalyDetectionSerializer
from .predict import predict_expenses, predict_loan_eligibility, predict_group_expenses, predict_anomalies

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

class GroupExpenseForecastView(APIView):
    def post(self, request):
        serializer = GroupExpenseForecastSerializer(data=request.data)
        if serializer.is_valid():
            prediction = predict_group_expenses(serializer.validated_data['data'])
            return Response({"prediction": prediction})
        return Response(serializer.errors, status=400)

class AnomalyDetectionView(APIView):
    def post(self, request):
        serializer = AnomalyDetectionSerializer(data=request.data)
        if serializer.is_valid():
            predictions = predict_anomalies(serializer.validated_data['data'])
            return Response({"predictions": predictions})
        return Response(serializer.errors, status=400)