from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import traceback

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get all 6 form data inputs
            age = request.form.get('age')
            sex = request.form.get('sex')
            bmi = request.form.get('bmi')
            children = request.form.get('children')
            smoker = request.form.get('smoker')
            region = request.form.get('region')
            
            # Validate all required fields
            if not all([age, sex, bmi, children, smoker, region]):
                return render_template('home.html', results="Please fill all required fields")
            
            # Create the data object with all 6 features
            data = CustomData(
                age=float(age),
                sex=sex,
                bmi=float(bmi),
                children=int(children),
                smoker=smoker,
                region=region
            )
            
            pred_df = data.get_data_as_data_frame()
            print("Input data:")
            print(pred_df)
            print(f"Data columns: {pred_df.columns.tolist()}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print(f"Predicted expenses: ${results[0]:.2f}")
            
            return render_template('home.html', results=f"${results[0]:.2f}")
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"Full error: {traceback.format_exc()}")
            
            # Provide more user-friendly error message
            if "expecting" in str(e) and "features" in str(e):
                error_message = "Model configuration error. Please check if the model was trained with the same features."
            elif "NoneType" in str(e):
                error_message = "Please fill all required fields."
            
            return render_template('home.html', results=error_message)

@app.route('/debug')
def debug_info():
    try:
        from src.utils import load_object
        
        # Load preprocessor and model to check their expected features
        preprocessor = load_object('artifacts/preprocessor.pkl')
        model = load_object('artifacts/model.pkl')
        
        info = {
            'model_expected_features': getattr(model, 'n_features_in_', 'Unknown'),
            'preprocessor_features': []
        }
        
        # Get feature names from preprocessor
        for name, transformer, columns in preprocessor.transformers:
            info['preprocessor_features'].extend(columns)
        
        return f"""
        <h1>Debug Information</h1>
        <p><b>Model expects:</b> {info['model_expected_features']} features</p>
        <p><b>Preprocessor handles:</b> {info['preprocessor_features']}</p>
        <p><b>Total features:</b> {len(info['preprocessor_features'])}</p>
        <p>Your form should have exactly these fields</p>
        """
        
    except Exception as e:
        return f"Debug error: {str(e)}"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)