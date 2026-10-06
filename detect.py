import sys
import argparse
import joblib
import numpy as np
import warnings
from milestone2_features import StylometricFeaturesExtractor

# Suppress warnings for cleaner interactive output
warnings.filterwarnings('ignore')

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
    parser = argparse.ArgumentParser(description="Test The Writing Detective model interactively.")
    parser.add_argument('--text', type=str, help="Text paragraph to analyze (in quotes).")
    args = parser.parse_args()
    
    text = args.text
    if not text:
        print("Enter a paragraph of text to analyze (Press Enter twice to submit):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        text = " ".join(lines)
        
    if not text.strip():
        print("No text provided. Exiting.")
        return
        
    try:
        pipeline = joblib.load('data/best_model_pipeline.pkl')
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
        print("Did you run Milestone 3?")
        return
        
    print("\n" + "="*50)
    print("WRITING DETECTIVE ANALYSIS")
    print("="*50)
    
    # 1. Prediction
    predicted_author = pipeline.predict([text])[0]
    print(f"\n=> PREDICTED AUTHOR: {predicted_author}\n")
    
    # 2. Probability / Confidence Score
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        print("=> CONFIDENCE SCORES:")
        # Sort scores descending
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            print(f"   {classes[idx]}: {probs[idx] * 100:.1f}%")
    else:
        print("=> CONFIDENCE SCORES:")
        print("   Model (e.g. SVM without proba) does not support probability estimates.")
        
    # 3. Top Stylistic Features
    extractor = pipeline.named_steps['features']
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    
    # Extract and scale features
    features = extractor.transform([text])
    scaled_features = scaler.transform(features)[0]
    feature_names = get_feature_names()
    
    print("\n=> TOP 3 DRIVING STYLISTIC FEATURES:")
    
    if hasattr(clf, 'coef_'):
        # Linear Models: Contribution = coefficient_for_class * scaled_feature_value
        class_idx = list(clf.classes_).index(predicted_author)
        coefs = clf.coef_[class_idx]
        contributions = coefs * scaled_features
        # Find the features that had the largest POSITIVE contribution pushing toward this class
        top_indices = np.argsort(contributions)[::-1][:3]
        
        for idx in top_indices:
            val = scaled_features[idx]
            direction = "Unusually high" if val > 0 else "Unusually low"
            clean_name = feature_names[idx].replace('_', ' ')
            print(f"   - {direction} {clean_name}")
            
    elif hasattr(clf, 'feature_importances_'):
        # Tree Models: We proxy this by multiplying global importance by the magnitude of the text's feature value.
        # This highlights important features that have extreme values in this specific text chunk.
        contributions = clf.feature_importances_ * np.abs(scaled_features)
        top_indices = np.argsort(contributions)[::-1][:3]
        
        for idx in top_indices:
            val = scaled_features[idx]
            direction = "Unusually high" if val > 0 else "Unusually low"
            clean_name = feature_names[idx].replace('_', ' ')
            print(f"   - {direction} {clean_name}")
    else:
        # Fallback: Just look at the most extreme standard-deviations (scaled features)
        contributions = np.abs(scaled_features)
        top_indices = np.argsort(contributions)[::-1][:3]
        for idx in top_indices:
            val = scaled_features[idx]
            direction = "Unusually high" if val > 0 else "Unusually low"
            clean_name = feature_names[idx].replace('_', ' ')
            print(f"   - {direction} {clean_name}")
            
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
