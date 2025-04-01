from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData', methods=['POST','GET'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_data)

        return render_template('form.html', results=results[0]//0.1)
    
if __name__ == "__main__":
    app.run(host=10000,debug=True)
