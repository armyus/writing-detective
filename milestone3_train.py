import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from milestone2_features import StylometricFeaturesExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    if not os.path.exists('data/dataset.csv'):
        logging.error("dataset.csv not found. Please run Milestone 1 first.")
        return
        
    logging.info("Loading dataset...")
    df = pd.read_csv('data/dataset.csv')
    
    # Train-test split (80% train, 20% test hold-out for Milestone 4)
    X = df['text'].values
    y = df['author'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Save test set for Milestone 4 evaluation
    joblib.dump((X_test, y_test), 'data/test_data.pkl')
    logging.info("Saved test dataset to data/test_data.pkl")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    best_score = 0
    best_model_name = ""
    
    logging.info("Evaluating models with Cross-Validation...")
    logging.info("(This might take a minute as text features are being extracted on the fly)")
    
    # We use a Pipeline for each to prevent data leakage (particularly for StandardScaler)
    for name, model in models.items():
        pipeline = Pipeline([
            ('features', StylometricFeaturesExtractor()),
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        
        # 3-fold CV
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
        mean_score = np.mean(scores)
        logging.info(f"{name} CV Accuracy: {mean_score:.4f} (+/- {np.std(scores):.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            
    logging.info(f"\nBest Model identified: {best_model_name}")
    
    # Tune the best model
    logging.info(f"Tuning {best_model_name} with GridSearchCV...")
    if best_model_name == 'Logistic Regression':
        param_grid = {'clf__C': [0.1, 1.0, 10.0]}
    elif best_model_name == 'SVM':
        param_grid = {'clf__C': [0.1, 1.0, 10.0], 'clf__kernel': ['linear', 'rbf']}
    else:  # Random Forest
        param_grid = {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 10, 20]}
        
    best_pipeline = Pipeline([
        ('features', StylometricFeaturesExtractor()),
        ('scaler', StandardScaler()),
        ('clf', models[best_model_name])
    ])
    
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best tuned CV accuracy: {grid_search.best_score_:.4f}")
    
    # Retrain on full training set using best parameters (GridSearchCV does this automatically)
    final_model = grid_search.best_estimator_
    
    # Save the fully trained pipeline
    joblib.dump(final_model, 'data/best_model_pipeline.pkl')
    logging.info("Saved final tuned model pipeline to data/best_model_pipeline.pkl")

if __name__ == "__main__":
    main()
