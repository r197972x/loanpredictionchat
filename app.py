from flask import Flask, request, render_template
from pyspark.ml import PipelineModel

app = Flask(__name__)
model = PipelineModel.load('loan_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    prediction = predict_loan_approval(data)
    return render_template('result.html', prediction=prediction)

def predict_loan_approval(data):
    # Preprocess the input data
    gender = 1 if data['gender'] == 'Male' else 0
    married = 1 if data['married'] == 'Yes' else 0
    dependents = int(data['dependents'])
    education = 1 if data['education'] == 'Graduate' else 0
    self_employed = 1 if data['self_employed'] == 'Yes' else 0
    applicant_income = int(data['applicant_income'])
    coapplicant_income = int(data['coapplicant_income'])
    loan_amount = int(data['loan_amount'])
    loan_amount_term = int(data['loan_amount_term'])
    credit_history = int(data['credit_history'])
    property_area = 1 if data['property_area'] == 'Urban' else 0
    
    # Run the PySpark model to predict loan approval
    prediction = model.predict([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])[0]
    return 'Approved' if prediction == 1.0 else 'Not Approved'

if __name__ == '__main__':
    app.run(debug=True)
