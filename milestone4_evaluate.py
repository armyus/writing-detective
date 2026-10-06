import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

from milestone2_features import StylometricFeaturesExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_feature_names():
    """Reconstructs the ordered list of feature names from the extractor."""
    extractor = StylometricFeaturesExtractor()
    names = []
    
    for p in extractor.punctuations:
        names.append(f'punct_{p}_density')
        
    names.append('avg_sentence_length')
    names.append('std_sentence_length')
    
    for fw in extractor.function_words:
        names.append(f'fw_{fw}_ratio')
        
    for pos in ['NN', 'VB', 'JJ', 'RB']:
        names.append(f'pos_{pos}_ratio')
        
    return names

def main():
    if not os.path.exists('data/test_data.pkl') or not os.path.exists('data/best_model_pipeline.pkl'):
        logging.error("Test data or model not found. Please run Milestone 3 first.")
        return
        
    logging.info("Loading test data and model pipeline...")
    X_test, y_test = joblib.load('data/test_data.pkl')
    pipeline = joblib.load('data/best_model_pipeline.pkl')
    
    # 1. Predictions & Classification Report
    logging.info("Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix
    os.makedirs('results', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.ylabel('True Author')
    plt.xlabel('Predicted Author')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    logging.info("Saved confusion matrix plot to results/confusion_matrix.png")
    plt.close()
    
    # 3. Feature Importances / Coefficients
    clf = pipeline.named_steps['clf']
    feature_names = get_feature_names()
    
    importances = None
    print("\n" + "="*50)
    if hasattr(clf, 'coef_'):
        # For linear models (Logistic Regression, Linear SVM)
        # We take the absolute mean across classes to find overall most impactful features
        importances = np.mean(np.abs(clf.coef_), axis=0)
        print("TOP 10 FEATURE COEFFICIENTS (Absolute Mean across classes)")
    elif hasattr(clf, 'feature_importances_'):
        # For tree-based models
        importances = clf.feature_importances_
        print("TOP 10 FEATURE IMPORTANCES")
    
    print("="*50)
        
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    else:
        print("Model does not expose feature_importances_ or coef_ (e.g., non-linear SVM).")
        
    # 4. PCA Scatter Plot
    logging.info("\nGenerating 2D PCA scatter plot...")
    
    # We extract and scale features manually to apply PCA
    extractor = pipeline.named_steps['features']
    scaler = pipeline.named_steps['scaler']
    
    logging.info("Extracting features from test data for PCA visualization...")
    X_test_features = extractor.transform(X_test)
    X_test_scaled = scaler.transform(X_test_features)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_test_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_test, palette='Set2', alpha=0.8, edgecolor=None)
    plt.title('PCA of Stylometric Features (Test Set)')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title='Author')
    plt.tight_layout()
    plt.savefig('results/pca_clusters.png')
    logging.info("Saved PCA scatter plot to results/pca_clusters.png")
    plt.close()
    
    print("\nPipeline execution complete. All outputs are saved in the 'results' folder.")

if __name__ == "__main__":
    main()
