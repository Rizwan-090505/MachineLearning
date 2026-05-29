"""
adaboost.py
AdaBoost classifier with numpy vectorization
Roll Number: BSAI23004
"""

import numpy as np
from decision_tree import DecisionTreeClassifier


class AdaBoostClassifier:
    """Adaptive Boosting classifier."""
    
    def __init__(self, n_estimators=50, learning_rate=1.0, base_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_depth = base_depth
        self.random_state = random_state
        self.estimators_ = []
        self.alphas_ = []
        self.estimator_errors_ = []
    
    def fit(self, X, y):
        """Fit AdaBoost ensemble."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        n_samples = len(X)
        
        # Initialize uniform weights (vectorized)
        weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.alphas_ = []
        self.estimator_errors_ = []
        
        for m in range(self.n_estimators):
            # Train weak learner
            tree = DecisionTreeClassifier(
                max_depth=self.base_depth,
                min_samples_split=1,
                random_state=self.random_state + m if self.random_state is not None else None
            )
            tree.fit(X, y)
            
            # Get predictions (vectorized)
            y_pred = tree.predict(X)
            
            # Calculate weighted error (vectorized)
            errors = (y_pred != y).astype(float)
            weighted_error = np.sum(weights * errors)
            
            # Clamp error to prevent log issues
            weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)
            
            # Calculate alpha
            alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            
            # Update weights (vectorized)
            weights *= np.exp(-alpha * y * (2 * y_pred - 1))
            weights /= np.sum(weights)  # Normalize
            
            self.estimators_.append(tree)
            self.alphas_.append(alpha)
            self.estimator_errors_.append(weighted_error)
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float32)
        
        # Get weighted sum of predictions (vectorized)
        score = np.zeros(len(X))
        
        for alpha, tree in zip(self.alphas_, self.estimators_):
            y_pred = tree.predict(X)
            score += alpha * (2 * y_pred - 1)
        
        return (score > 0).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X, dtype=np.float32)
        
        score = np.zeros(len(X))
        for alpha, tree in zip(self.alphas_, self.estimators_):
            y_pred = tree.predict(X)
            score += alpha * (2 * y_pred - 1)
        
        # Convert score to probability using sigmoid-like transform
        proba_1 = 1.0 / (1.0 + np.exp(-2 * score))
        proba_0 = 1.0 - proba_1
        
        return np.column_stack([proba_0, proba_1])
    
    def staged_score(self, X, y):
        """Get accuracy at each boosting round (vectorized)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        scores = []
        score = np.zeros(len(X))
        
        for alpha, tree in zip(self.alphas_, self.estimators_):
            y_pred = tree.predict(X)
            score += alpha * (2 * y_pred - 1)
            
            predictions = (score > 0).astype(int)
            accuracy = np.mean(predictions == y)
            scores.append(accuracy)
        
        return scores
    
    def staged_predict(self, X):
        """Get predictions at each boosting round."""
        X = np.asarray(X, dtype=np.float32)
        
        predictions = []
        score = np.zeros(len(X))
        
        for alpha, tree in zip(self.alphas_, self.estimators_):
            y_pred = tree.predict(X)
            score += alpha * (2 * y_pred - 1)
            predictions.append((score > 0).astype(int).copy())
        
        return predictions
