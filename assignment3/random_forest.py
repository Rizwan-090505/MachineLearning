"""
random_forest.py
Random Forest with numpy vectorization and feature subsampling
Roll Number: BSAI23004
"""

import numpy as np
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest classifier with feature subsampling."""
    
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []
        self.oob_indices_ = []
        self.n_features_ = None
    
    def _get_n_features(self, n_features):
        """Calculate number of features to sample."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:  # None or 'all'
            return n_features
    
    def _bootstrap_sample(self, X, y, random_state):
        """Generate bootstrap sample."""
        n = len(X)
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[indices] = False
        return X[indices], y[indices], oob_mask, indices
    
    def fit(self, X, y):
        """Fit random forest."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        self.n_features_ = X.shape[1]
        n_features_to_sample = self._get_n_features(self.n_features_)
        
        rng = np.random.RandomState(self.random_state)
        
        self.estimators_ = []
        self.feature_indices_ = []
        self.oob_indices_ = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot, oob_mask, indices = self._bootstrap_sample(
                X, y, 
                self.random_state + i if self.random_state is not None else None
            )
            
            # Random feature subset
            feature_idx = rng.choice(self.n_features_, size=n_features_to_sample, replace=False)
            X_boot_sub = X_boot[:, feature_idx]
            X_sub = X[:, feature_idx]
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=2,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            tree.fit(X_boot_sub, y_boot)
            
            self.estimators_.append(tree)
            self.feature_indices_.append(feature_idx)
            self.oob_indices_.append(np.where(oob_mask)[0])
        
        return self
    
    def predict(self, X):
        """Predict using majority voting."""
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            X_sub = X[:, feat_idx]
            preds = tree.predict(X_sub)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        
        # Majority vote (vectorized)
        votes = np.sum(predictions == 1, axis=0)
        return (votes > len(self.estimators_) / 2).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            X_sub = X[:, feat_idx]
            preds = tree.predict(X_sub)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        votes = np.sum(predictions == 1, axis=0)
        proba = votes / len(self.estimators_)
        
        return np.column_stack([1 - proba, proba])
    
    def oob_score(self, X, y):
        """Calculate OOB accuracy (vectorized)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        oob_preds = np.zeros((len(y), 2))
        oob_counts = np.zeros(len(y))
        
        for i, (tree, feat_idx) in enumerate(zip(self.estimators_, self.feature_indices_)):
            oob_idx = self.oob_indices_[i]
            if len(oob_idx) > 0:
                X_oob = X[oob_idx][:, feat_idx]
                preds = tree.predict(X_oob)
                oob_preds[oob_idx, preds] += 1
                oob_counts[oob_idx] += 1
        
        # Get predictions for samples with OOB estimates
        valid_mask = oob_counts > 0
        if np.sum(valid_mask) == 0:
            return 0.0
        
        oob_predictions = np.argmax(oob_preds[valid_mask], axis=1)
        accuracy = np.mean(oob_predictions == y[valid_mask])
        
        return accuracy
    
    def get_tree_predictions(self, X):
        """Get predictions from all trees (for diversity analysis)."""
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            X_sub = X[:, feat_idx]
            preds = tree.predict(X_sub)
            predictions.append(preds)
        
        return np.array(predictions)  # Shape: (n_trees, n_samples)
    
    def feature_usage_counts(self):
        """Count feature usage across trees (vectorized)."""
        counts = np.zeros(self.n_features_)
        
        for feat_idx in self.feature_indices_:
            counts[feat_idx] += 1
        
        return counts
