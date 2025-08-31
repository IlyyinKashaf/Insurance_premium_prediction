# train_pipeline.py
import os
import sys
sys.path.append('.')

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        print("Starting the training pipeline...")
        
        # 1. Data Ingestion
        print("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_ingestion()
        print(f"Train data path: {train_data_path}")
        print(f"Test data path: {test_data_path}")
        
        # 2. Data Transformation
        print("Step 2: Data Transformation")
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print(f"Train array shape: {train_array.shape}")
        print(f"Test array shape: {test_array.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
        # 3. Model Training
        print("Step 3: Model Training")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_array, test_array)
        
        print("\n=== TRAINING COMPLETED ===")
        print(f"Best model: {result['model_name']}")
        print(f"R2 score: {result['model_score']:.4f}")
        print(f"Best CV score: {result['best_score']:.4f}")
        print(f"Number of features used: {result['num_features']}")
        print(f"Model saved at: {result['model_path']}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()