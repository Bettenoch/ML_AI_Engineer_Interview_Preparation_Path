import hashlib
import numpy as np
from collections import defaultdict
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class FeatureHasher:
    def __init__(self, n_features=2**18, hash_func='md5'):
        """
        Feature hasher for text data.
        
        Args:
            n_features: Size of feature space (should be power of 2)
            hash_func: Hash function to use ('md5', 'sha1', or callable)
        """
        self.n_features = n_features
        self.hash_func = hash_func
        
    def _hash_feature(self, feature):
        """Hash a feature string to an index."""
        if callable(self.hash_func):
            hash_val = self.hash_func(feature.encode('utf-8'))
        else:
            hasher = hashlib.new(self.hash_func)
            hasher.update(feature.encode('utf-8'))
            hash_val = int(hasher.hexdigest(), 16)
        
        return hash_val % self.n_features
    
    def _signed_hash(self, feature):
        """Generate signed hash to reduce collisions' impact."""
        # Use different hash for sign
        sign_hasher = hashlib.md5()
        sign_hasher.update(('sign_' + feature).encode('utf-8'))
        sign = 1 if int(sign_hasher.hexdigest(), 16) % 2 == 0 else -1
        return sign
    
    def transform(self, texts, use_signed_hash=True):
        """
        Transform texts to hashed feature vectors.
        
        Args:
            texts: List of text documents
            use_signed_hash: Whether to use signed hashing
            
        Returns:
            Sparse feature matrix
        """
        n_samples = len(texts)
        feature_matrix = np.zeros((n_samples, self.n_features))
        
        for i, text in enumerate(texts):
            features = self._extract_features(text)
            feature_counts = defaultdict(float)
            
            for feature, count in features.items():
                hash_idx = self._hash_feature(feature)
                if use_signed_hash:
                    sign = self._signed_hash(feature)
                    feature_counts[hash_idx] += sign * count
                else:
                    feature_counts[hash_idx] += count
            
            for idx, count in feature_counts.items():
                feature_matrix[i, idx] = count
                
        return feature_matrix
    
    def _extract_features(self, text):
        """Extract features from text (unigrams, bigrams, etc.)."""
        features = defaultdict(int)
        
        # Preprocessing
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Unigrams
        for word in words:
            features[f'unigram_{word}'] += 1
        
        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            features[f'bigram_{bigram}'] += 1
        
        # Character n-grams (useful for handling typos)
        for word in words:
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                features[f'char3_{trigram}'] += 1
        
        return features

class HashedTextClassifier:
    def __init__(self, n_features=2**18, classifier=None):
        """
        Text classifier using feature hashing.
        
        Args:
            n_features: Size of hashed feature space
            classifier: Sklearn-compatible classifier
        """
        self.hasher = FeatureHasher(n_features=n_features)
        self.classifier = classifier or LogisticRegression(max_iter=1000)
        
    def fit(self, texts, labels):
        """Train the classifier."""
        X = self.hasher.transform(texts)
        self.classifier.fit(X, labels)
        return self
    
    def predict(self, texts):
        """Make predictions on new texts."""
        X = self.hasher.transform(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Get prediction probabilities."""
        X = self.hasher.transform(texts)
        return self.classifier.predict_proba(X)

# Example usage and demonstration
def demo_feature_hashing():
    # Sample data
    train_texts = [
        "I love this movie, it's fantastic!",
        "This film is terrible, waste of time",
        "Great acting and wonderful story",
        "Boring and poorly written script",
        "Amazing cinematography and direction",
        "Awful dialogue and bad acting"
    ]
    
    train_labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    test_texts = [
        "This movie is absolutely wonderful!",
        "Very disappointing and boring film"
    ]
    
    # Train classifier
    classifier = HashedTextClassifier(n_features=2**12)  # Smaller for demo
    classifier.fit(train_texts, train_labels)
    
    # Make predictions
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    print("Feature Hashing Text Classification Demo")
    print("=" * 40)
    print(f"Feature space size: {classifier.hasher.n_features}")
    print()
    
    for i, text in enumerate(test_texts):
        pred = predictions[i]
        prob = probabilities[i]
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(prob)
        
        print(f"Text: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print()

# Advanced feature hasher with TF-IDF weighting
class TFIDFFeatureHasher(FeatureHasher):
    def __init__(self, n_features=2**18, max_df=0.95, min_df=2):
        super().__init__(n_features)
        self.max_df = max_df
        self.min_df = min_df
        self.doc_freq = defaultdict(int)
        self.n_docs = 0
        
    def fit(self, texts):
        """Fit IDF weights on training data."""
        self.n_docs = len(texts)
        feature_doc_count = defaultdict(set)
        
        for i, text in enumerate(texts):
            features = self._extract_features(text)
            for feature in features:
                feature_doc_count[feature].add(i)
        
        # Calculate document frequencies
        for feature, docs in feature_doc_count.items():
            self.doc_freq[feature] = len(docs)
        
        return self
    
    def transform(self, texts, use_tfidf=True):
        """Transform with optional TF-IDF weighting."""
        if not use_tfidf:
            return super().transform(texts)
            
        n_samples = len(texts)
        feature_matrix = np.zeros((n_samples, self.n_features))
        
        for i, text in enumerate(texts):
            features = self._extract_features(text)
            feature_weights = defaultdict(float)
            doc_length = sum(features.values())
            
            for feature, tf in features.items():
                # TF-IDF calculation
                tf_norm = tf / doc_length
                df = self.doc_freq.get(feature, 1)
                idf = np.log(self.n_docs / df)
                tfidf = tf_norm * idf
                
                hash_idx = self._hash_feature(feature)
                sign = self._signed_hash(feature)
                feature_weights[hash_idx] += sign * tfidf
            
            for idx, weight in feature_weights.items():
                feature_matrix[i, idx] = weight
                
        return feature_matrix

if __name__ == "__main__":
    demo_feature_hashing()