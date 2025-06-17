import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implemented from scratch with optional Laplace smoothing
    """
    
    def __init__(self, alpha=0.0):
        """
        Initialize the classifier
        
        Parameters:
        alpha (float): Laplace smoothing parameter (0 = no smoothing, 1 = add-one smoothing)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
        self.vocabulary = set()
        self.vocab_size = 0
        
    def _tokenize(self, text):
        """Simple tokenization function"""
        if isinstance(text, str):
            # Convert to lowercase and extract words
            tokens = re.findall(r'\b\w+\b', text.lower())
            return tokens
        return []
    
    def _calculate_class_priors(self, y):
        """Calculate prior probabilities for each class"""
        class_counts = Counter(y)
        total_samples = len(y)
        
        for class_label in class_counts:
            self.class_priors[class_label] = class_counts[class_label] / total_samples
    
    def _calculate_feature_probabilities(self, X, y):
        """Calculate feature probabilities for each class"""
        # Initialize feature probability dictionaries
        for class_label in self.classes:
            self.feature_probs[class_label] = defaultdict(float)
        
        # Count feature occurrences for each class
        class_feature_counts = {class_label: defaultdict(int) for class_label in self.classes}
        class_total_features = {class_label: 0 for class_label in self.classes}
        
        for i, class_label in enumerate(y):
            for feature in X[i]:
                class_feature_counts[class_label][feature] += 1
                class_total_features[class_label] += 1
        
        # Calculate probabilities with optional Laplace smoothing
        for class_label in self.classes:
            for feature in self.vocabulary:
                feature_count = class_feature_counts[class_label][feature]
                total_features = class_total_features[class_label]
                
                # Apply Laplace smoothing: P(feature|class) = (count + alpha) / (total + alpha * vocab_size)
                self.feature_probs[class_label][feature] = (
                    (feature_count + self.alpha) / 
                    (total_features + self.alpha * self.vocab_size)
                )
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Parameters:
        X: List of documents (strings) or list of tokenized documents
        y: List of class labels
        """
        self.classes = list(set(y))
        
        # Tokenize documents if they're strings
        if X and isinstance(X[0], str):
            X_tokenized = [self._tokenize(doc) for doc in X]
        else:
            X_tokenized = X
        
        # Build vocabulary
        for doc in X_tokenized:
            self.vocabulary.update(doc)
        
        self.vocab_size = len(self.vocabulary)
        
        # Calculate priors and feature probabilities
        self._calculate_class_priors(y)
        self._calculate_feature_probabilities(X_tokenized, y)
        
        return self
    
    def _predict_single(self, document):
        """Predict class for a single document"""
        # Tokenize if necessary
        if isinstance(document, str):
            tokens = self._tokenize(document)
        else:
            tokens = document
        
        class_scores = {}
        
        for class_label in self.classes:
            # Start with log prior probability
            score = np.log(self.class_priors[class_label])
            
            # Add log probabilities for each feature
            for token in tokens:
                if token in self.vocabulary:
                    # Add log probability to avoid underflow
                    score += np.log(self.feature_probs[class_label][token])
                else:
                    # Handle unseen words with smoothing
                    if self.alpha > 0:
                        score += np.log(self.alpha / (sum(self.feature_probs[class_label].values()) + 
                                                    self.alpha * self.vocab_size))
            
            class_scores[class_label] = score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X):
        """
        Predict classes for multiple documents
        
        Parameters:
        X: List of documents (strings) or list of tokenized documents
        
        Returns:
        List of predicted class labels
        """
        return [self._predict_single(doc) for doc in X]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for documents
        
        Parameters:
        X: List of documents
        
        Returns:
        Dictionary with probabilities for each class
        """
        probabilities = []
        
        for document in X:
            # Tokenize if necessary
            if isinstance(document, str):
                tokens = self._tokenize(document)
            else:
                tokens = document
            
            class_scores = {}
            
            for class_label in self.classes:
                score = np.log(self.class_priors[class_label])
                
                for token in tokens:
                    if token in self.vocabulary:
                        score += np.log(self.feature_probs[class_label][token])
                    else:
                        if self.alpha > 0:
                            score += np.log(self.alpha / (sum(self.feature_probs[class_label].values()) + 
                                                        self.alpha * self.vocab_size))
                
                class_scores[class_label] = score
            
            # Convert log scores to probabilities
            max_score = max(class_scores.values())
            exp_scores = {k: np.exp(v - max_score) for k, v in class_scores.items()}
            total = sum(exp_scores.values())
            probs = {k: v / total for k, v in exp_scores.items()}
            
            probabilities.append(probs)
        
        return probabilities

def create_sample_dataset():
    """Create a simple sample dataset for demonstration"""
    documents = [
        "I love this movie it's amazing",
        "This film is terrible and boring",
        "Great acting and wonderful story",
        "Awful movie waste of time",
        "Excellent cinematography and direction",
        "Poor script and bad acting",
        "Fantastic movie highly recommended",
        "Disappointing and uninteresting",
        "Beautiful visuals and great music",
        "Horrible film avoid at all costs"
    ]
    
    labels = ['positive', 'negative', 'positive', 'negative', 'positive', 
              'negative', 'positive', 'negative', 'positive', 'negative']
    
    return documents, labels

def load_newsgroup_data():
    """Load a subset of 20 newsgroups data for text classification"""
    print("Loading 20 newsgroups dataset...")
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                        shuffle=True, random_state=42,
                                        remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, 
                                       shuffle=True, random_state=42,
                                       remove=('headers', 'footers', 'quotes'))
    
    return (newsgroups_train.data, newsgroups_train.target, 
            newsgroups_test.data, newsgroups_test.target, 
            newsgroups_train.target_names)

def evaluate_classifier(y_true, y_pred, class_names=None):
    """Evaluate classifier performance"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return accuracy

def compare_smoothing_effects():
    """Compare classifier performance with and without Laplace smoothing"""
    print("=== COMPARING LAPLACE SMOOTHING EFFECTS ===\n")
    
    # Test on sample dataset
    print("1. Testing on Sample Movie Review Dataset:")
    print("-" * 50)
    
    documents, labels = create_sample_dataset()
    X_train, X_test, y_train, y_test = train_test_split(documents, labels, 
                                                        test_size=0.3, random_state=42)
    
    # Without smoothing
    print("Without Laplace Smoothing (alpha=0):")
    nb_no_smooth = NaiveBayesClassifier(alpha=0.0)
    nb_no_smooth.fit(X_train, y_train)
    y_pred_no_smooth = nb_no_smooth.predict(X_test)
    acc_no_smooth = evaluate_classifier(y_test, y_pred_no_smooth)
    
    print("\n" + "="*50 + "\n")
    
    # With smoothing
    print("With Laplace Smoothing (alpha=1):")
    nb_smooth = NaiveBayesClassifier(alpha=1.0)
    nb_smooth.fit(X_train, y_train)
    y_pred_smooth = nb_smooth.predict(X_test)
    acc_smooth = evaluate_classifier(y_test, y_pred_smooth)
    
    print(f"\nSmoothing improved accuracy by: {acc_smooth - acc_no_smooth:.4f}")
    
    return nb_no_smooth, nb_smooth

def test_on_newsgroups():
    """Test classifier on 20 newsgroups dataset"""
    print("\n" + "="*60)
    print("2. Testing on 20 Newsgroups Dataset:")
    print("="*60)
    
    X_train, y_train, X_test, y_test, class_names = load_newsgroup_data()
    
    # Limit dataset size for faster processing
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:200]
    y_test = y_test[:200]
    
    print(f"Training on {len(X_train)} documents, testing on {len(X_test)} documents")
    print(f"Classes: {class_names}\n")
    
    # Test with different smoothing values
    smoothing_values = [0.0, 0.1, 1.0, 2.0]
    results = {}
    
    for alpha in smoothing_values:
        print(f"Testing with alpha = {alpha}:")
        nb = NaiveBayesClassifier(alpha=alpha)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        accuracy = evaluate_classifier(y_test, y_pred, class_names)
        results[alpha] = accuracy
        print("-" * 40)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    alphas = list(results.keys())
    accuracies = list(results.values())
    
    plt.plot(alphas, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Laplace Smoothing Parameter (alpha)')
    plt.ylabel('Accuracy')
    plt.title('Effect of Laplace Smoothing on Classifier Performance')
    plt.grid(True, alpha=0.3)
    plt.xticks(alphas)
    for i, (alpha, acc) in enumerate(zip(alphas, accuracies)):
        plt.annotate(f'{acc:.3f}', (alpha, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    plt.show()
    
    return results

def compare_with_sklearn():
    """Compare our implementation with sklearn's MultinomialNB"""
    print("\n" + "="*60)
    print("3. Comparison with Scikit-learn Implementation:")
    print("="*60)
    
    X_train, y_train, X_test, y_test, class_names = load_newsgroup_data()
    
    # Limit dataset size
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:200]
    y_test = y_test[:200]
    
    # Vectorize the text data for sklearn
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Our implementation
    print("Our Naive Bayes Implementation:")
    our_nb = NaiveBayesClassifier(alpha=1.0)
    our_nb.fit(X_train, y_train)
    our_pred = our_nb.predict(X_test)
    our_accuracy = evaluate_classifier(y_test, our_pred, class_names)
    
    print("\n" + "-"*40 + "\n")
    
    # Sklearn implementation
    print("Scikit-learn MultinomialNB:")
    sklearn_nb = MultinomialNB(alpha=1.0)
    sklearn_nb.fit(X_train_vec, y_train)
    sklearn_pred = sklearn_nb.predict(X_test_vec)
    sklearn_accuracy = evaluate_classifier(y_test, sklearn_pred, class_names)
    
    print(f"\nAccuracy Comparison:")
    print(f"Our implementation: {our_accuracy:.4f}")
    print(f"Scikit-learn:       {sklearn_accuracy:.4f}")
    print(f"Difference:         {abs(our_accuracy - sklearn_accuracy):.4f}")

def demonstrate_classifier():
    """Demonstrate the classifier with examples"""
    print("\n" + "="*60)
    print("4. Classifier Demonstration:")
    print("="*60)
    
    # Train on sample data
    documents, labels = create_sample_dataset()
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(documents, labels)
    
    # Test examples
    test_examples = [
        "This movie is absolutely fantastic and amazing",
        "Terrible acting and horrible plot",
        "Great film highly recommend",
        "Waste of time boring movie"
    ]
    
    print("Test Examples and Predictions:")
    print("-" * 40)
    
    for example in test_examples:
        prediction = nb.predict([example])[0]
        probabilities = nb.predict_proba([example])[0]
        
        print(f"Text: '{example}'")
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")
        print("-" * 40)

if __name__ == "__main__":
    print("NAIVE BAYES CLASSIFIER IMPLEMENTATION")
    print("="*60)
    
    # Step 1: Compare smoothing effects
    nb_no_smooth, nb_smooth = compare_smoothing_effects()
    
    # Step 2: Test on larger dataset
    newsgroup_results = test_on_newsgroups()
    
    # Step 3: Compare with sklearn
    compare_with_sklearn()
    
    # Step 4: Demonstrate classifier
    demonstrate_classifier()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION COMPLETE!")
    print("="*60)