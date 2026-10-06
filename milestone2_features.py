import string
import numpy as np
import pandas as pd
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

class StylometricFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # List of common function words
        self.function_words = [
            'the', 'and', 'of', 'to', 'a', 'in', 'it', 'is', 'was', 'i', 
            'for', 'that', 'you', 'he', 'be', 'with', 'on', 'at', 'by', 'which'
        ]
        
        # Punctuation marks we care about
        self.punctuations = [',', '.', ';', ':', '?', '!']
        
    def fit(self, X, y=None):
        # Nothing to fit for these static features
        return self
        
    def transform(self, X, y=None):
        features_list = []
        
        for text in X:
            features = {}
            
            # 1. Punctuation density (count per 1000 characters)
            text_length = len(text) if len(text) > 0 else 1
            for p in self.punctuations:
                features[f'punct_{p}_density'] = (text.count(p) / text_length) * 1000
                
            # Tokenize into sentences and words
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text.lower())
            
            # 2. Sentence stats
            num_sentences = len(sentences) if len(sentences) > 0 else 1
            words_per_sentence = [len(nltk.word_tokenize(s)) for s in sentences]
            features['avg_sentence_length'] = np.mean(words_per_sentence)
            features['std_sentence_length'] = np.std(words_per_sentence)
            
            # 3. Function words frequency (count per 100 words)
            num_words = len(words) if len(words) > 0 else 1
            word_freq = pd.Series(words).value_counts()
            for fw in self.function_words:
                count = word_freq.get(fw, 0)
                features[f'fw_{fw}_ratio'] = (count / num_words) * 100
                
            # 4. POS-tag ratios
            pos_tags = nltk.pos_tag(words)
            pos_counts = {
                'NN': 0, # Nouns
                'VB': 0, # Verbs
                'JJ': 0, # Adjectives
                'RB': 0  # Adverbs
            }
            
            for _, tag in pos_tags:
                for pos_prefix in pos_counts.keys():
                    if tag.startswith(pos_prefix):
                        pos_counts[pos_prefix] += 1
                        
            for pos, count in pos_counts.items():
                features[f'pos_{pos}_ratio'] = (count / num_words) * 100
                
            features_list.append(features)
            
        return pd.DataFrame(features_list).values

def main():
    import os
    
    if not os.path.exists('data/dataset.csv'):
        print("Error: data/dataset.csv not found. Please run Milestone 1 first.")
        return
        
    print("Loading dataset...")
    df = pd.read_csv('data/dataset.csv')
    
    # We will test on a tiny subset to show it works
    sample_texts = df['text'].head(5).tolist()
    
    print("Extracting features for 5 sample texts...")
    extractor = StylometricFeaturesExtractor()
    feature_matrix = extractor.transform(sample_texts)
    
    print(f"\nFeature matrix shape for 5 samples: {feature_matrix.shape}")
    print("\nSample feature row 0 (truncated to first 10 features):")
    print(feature_matrix[0][:10])
    
if __name__ == "__main__":
    main()
