# ml_app/serializers.py
from rest_framework import serializers

class ExpenseForecastSerializer(serializers.Serializer):
    data = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField()
        )
    )


class LoanEligibilitySerializer(serializers.Serializer):
    Gender = serializers.ChoiceField(choices=['Male', 'Female'])
    Married = serializers.ChoiceField(choices=['Yes', 'No'])
    Dependents = serializers.IntegerField()
    Education = serializers.ChoiceField(choices=['Graduate', 'Not Graduate'])
    Self_Employed = serializers.ChoiceField(choices=['Yes', 'No'])
    ApplicantIncome = serializers.FloatField()
    CoapplicantIncome = serializers.FloatField()
    LoanAmount = serializers.FloatField()
    Loan_Amount_Term = serializers.FloatField()
    Credit_History = serializers.IntegerField()
    Property_Area = serializers.ChoiceField(choices=['Urban', 'Semiurban', 'Rural'])


class GroupExpenseForecastSerializer(serializers.Serializer):
    data = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField()
        )
    )

