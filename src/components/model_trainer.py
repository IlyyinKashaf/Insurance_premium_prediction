import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("ModelTrainer initialized")

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        """
        Evaluate multiple models using GridSearchCV and return their performance scores
        """
        try:
            report = {}
            
            for model_name, model in models.items():
                try:
                    para = param.get(model_name, {})
                    
                    logging.info(f"Training {model_name}...")
                    
                    if para:  # If parameters exist, perform GridSearch
                        logging.info(f"Performing GridSearch for {model_name}")
                        gs = GridSearchCV(model, para, cv=3, scoring='r2', n_jobs=-1, verbose=0)
                        gs.fit(X_train, y_train)
                        
                        # Set the best parameters
                        model.set_params(**gs.best_params_)
                        logging.info(f"Best params for {model_name}: {gs.best_params_}")
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Make predictions and calculate score
                    y_test_pred = model.predict(X_test)
                    test_model_score = r2_score(y_test, y_test_pred)
                    
                    report[model_name] = test_model_score
                    logging.info(f"{model_name} R2 score: {test_model_score:.4f}")
                    
                except Exception as model_error:
                    logging.error(f"Error training {model_name}: {str(model_error)}")
                    report[model_name] = -999  # Very low score for failed models
                    continue
                
            return report
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Check if arrays are valid
            if train_array is None or test_array is None:
                raise CustomException("Train or test array is None", sys)
            
            if train_array.shape[0] == 0 or test_array.shape[0] == 0:
                raise CustomException("Train or test array is empty", sys)
                
            # Split into features (all columns except last) and target (last column - expenses)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All features: age, sex, bmi, children, smoker, region
                train_array[:, -1],   # Target: expenses
                test_array[:, :-1],   # All features: age, sex, bmi, children, smoker, region
                test_array[:, -1]     # Target: expenses
            )
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Testing data shape: {X_test.shape}")
            logging.info(f"Number of features: {X_train.shape[1]}")
            
            # Skip NaN check since data transformation already handles missing values
            
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 9],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                },
                "CatBoosting Regressor": {
                    'iterations': [50, 100, 200],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001]
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning...")
            model_report = self.evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            logging.info(f"Model evaluation completed. Results: {model_report}")
            
            # Filter out failed models (negative scores)
            valid_scores = {k: v for k, v in model_report.items() if v > -900}
            
            if not valid_scores:
                raise CustomException("All models failed to train", sys)
            
            # Get best model
            best_model_name = max(valid_scores, key=valid_scores.get)
            best_model_score = valid_scores[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                logging.warning(f"Best model score ({best_model_score:.4f}) is below threshold of 0.6")
                # You can choose to proceed anyway or raise an exception
                # raise CustomException("No best model found with sufficient accuracy")

            # Retrain the best model on full training data
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Verify the model was saved
            if os.path.exists(self.model_trainer_config.trained_model_file_path):
                logging.info(f"Model successfully saved at: {self.model_trainer_config.trained_model_file_path}")
            else:
                raise CustomException("Model file was not created successfully")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Final R2 score on test data: {r2_square:.4f}")
            
            return {
                'model_name': best_model_name,
                'model_score': r2_square,
                'best_score': best_model_score,
                'model_path': self.model_trainer_config.trained_model_file_path,
                'num_features': X_train.shape[1]
            }

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
        



