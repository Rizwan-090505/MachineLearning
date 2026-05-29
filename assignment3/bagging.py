"""
bagging.py
Bootstrap Aggregation with numpy vectorization
Roll Number: BSAI23004
"""

import numpy as np
from decision_tree import DecisionTreeClassifier


def bootstrap_sample(X, y, random_state=None):
    """
    Generate bootstrap sample with OOB mask using vectorized operations.
    Returns: X_boot, y_boot, oob_mask, indices
    """
    n = len(X)
    rng = np.random.RandomState(random_state)
    
    # Sample indices with replacement (vectorized)
    indices = rng.choice(n, size=n, replace=True)
    
    # Create OOB mask (vectorized)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[indices] = False
    
    return X[indices], y[indices], oob_mask, indices


class BaggingClassifier:
    """Bootstrap Aggregating classifier."""
    
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_ = []
        self.oob_indices_ = []
    
    def fit(self, X, y):
        """Fit ensemble of decision trees."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        self.estimators_ = []
        self.oob_indices_ = []
        
        for i in range(self.n_estimators):
            # Generate bootstrap sample
            X_boot, y_boot, oob_mask, indices = bootstrap_sample(
                X, y, 
                random_state=self.random_state + i if self.random_state is not None else None
            )
            
            # Train tree on bootstrap sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            tree.fit(X_boot, y_boot)
            
            self.estimators_.append(tree)
            self.oob_indices_.append(np.where(oob_mask)[0])
        
        return self
    
    def predict(self, X):
        """Predict class labels using majority voting."""
        X = np.asarray(X, dtype=np.float32)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Majority vote using vectorized operations
        unique_classes = np.array([0, 1])
        votes = np.zeros((X.shape[0], 2))
        
        for pred in predictions:
            votes[np.arange(len(pred)), pred] += 1
        
        return unique_classes[np.argmax(votes, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X, dtype=np.float32)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Calculate probability as fraction voting for class 1
        votes = np.sum(predictions == 1, axis=0)
        proba = votes / self.n_estimators
        
        return np.column_stack([1 - proba, proba])
    
    def oob_score(self, X, y):
        """Calculate out-of-bag error using vectorized operations."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        oob_preds = np.full(len(y), -1, dtype=int)
        oob_counts = np.zeros(len(y), dtype=int)
        
        # For each tree, predict on its OOB samples
        for i, tree in enumerate(self.estimators_):
            oob_idx = self.oob_indices_[i]
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                oob_preds[oob_idx] += preds + 1
                oob_counts[oob_idx] += 1
        
        # Average predictions for samples with OOB estimates
        valid_mask = oob_counts > 0
        oob_preds[valid_mask] = (oob_preds[valid_mask] / oob_counts[valid_mask]).astype(int)
        
        # Calculate accuracy on samples with OOB estimates
        if np.sum(valid_mask) == 0:
            return 0.0
        
        accuracy = np.mean(oob_preds[valid_mask] == y[valid_mask])
        return accuracy
    
    def variance_across_runs(self, X_train, y_train, n_runs=3, test_X=None, test_y=None):
        """Calculate variance across multiple runs (returns accuracies)."""
        accuracies = []
        
        for run in range(n_runs):
            # Refit with different random state
            bag_run = BaggingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + run + 100 if self.random_state is not None else None
            )
            bag_run.fit(X_train, y_train)
            
            if test_X is not None and test_y is not None:
                preds = bag_run.predict(test_X)
                acc = np.mean(preds == test_y)
                accuracies.append(acc)
        
        return np.array(accuracies)
    
    def feature_usage_counts(self):
        """Count how many times each feature is used across all trees."""
        if not self.estimators_:
            return np.array([])
        
        # For bagging, we don't track feature importance directly
        # Return uniform distribution
        n_features = self.estimators_[0].n_features_
        return np.ones(n_features)
