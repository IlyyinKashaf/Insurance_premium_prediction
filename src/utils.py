import os
import sys
import pickle
import logging  # Use built-in logging module
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object successfully saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        logging.info("Starting model evaluation...")

        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                
                # Get parameters for this model, empty dict if not found
                para = param.get(model_name, {})
                
                if para:  # If parameters exist, perform GridSearch
                    logging.info(f"Performing GridSearch for {model_name}")
                    gs = GridSearchCV(model, para, cv=3, scoring='r2', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                    logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
                
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score
                logging.info(f"{model_name} R2 score: {test_model_score:.4f}")

            except Exception as model_error:
                logging.error(f"Error evaluating {model_name}: {str(model_error)}")
                report[model_name] = -1  # Assign a negative score for failed models
                continue

        return report

    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)