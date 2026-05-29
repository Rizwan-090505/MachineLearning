"""
decision_tree.py
Fast Decision Tree Classifier with numpy vectorization
Roll Number: BSAI23004
"""

import numpy as np


def entropy(y):
    """Calculate entropy with numpy vectorization."""
    n = len(y)
    if n == 0:
        return 0.0
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y, mask):
    """Vectorized information gain calculation."""
    n = len(y)
    left_y, right_y = y[mask], y[~mask]
    
    parent_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)
    
    left_weight = len(left_y) / n
    right_weight = len(right_y) / n
    
    return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)


class TreeNode:
    """Simple tree node for decision tree."""
    __slots__ = ['is_leaf', 'prediction', 'feature', 'threshold', 'left', 'right']
    
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    """Fast decision tree with depth limits for ensemble use."""
    
    def __init__(self, max_depth=5, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root_ = None
        self.n_features_ = None
    
    def fit(self, X, y):
        """Fit tree with numpy arrays."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, depth=0)
        return self
    
    def _leaf(self, y):
        """Create leaf node with majority class."""
        unique, counts = np.unique(y, return_counts=True)
        prediction = unique[np.argmax(counts)]
        return TreeNode(is_leaf=True, prediction=prediction)
    
    def _best_split(self, X, y):
        """Find best split using vectorized operations."""
        n_features = X.shape[1]
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            col = X[:, feature]
            thresholds = np.unique(col)
            
            if len(thresholds) < 2:
                continue
            
            # Use midpoints between unique values as thresholds
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            # Limit number of thresholds for speed
            if len(thresholds) > 20:
                idx = np.linspace(0, len(thresholds) - 1, 20, dtype=int)
                thresholds = thresholds[idx]
            
            # Vectorized threshold evaluation
            for threshold in thresholds:
                mask = col <= threshold
                n_left = np.sum(mask)
                n_right = len(mask) - n_left
                
                if n_left < 1 or n_right < 1:
                    continue
                
                gain = information_gain(y, mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build(self, X, y, depth):
        """Recursively build tree with depth limit."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stop conditions
        if n_samples < self.min_samples_split or n_classes == 1:
            return self._leaf(y)
        
        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf(y)
        
        # Find best split
        feature, threshold, gain = self._best_split(X, y)
        
        if feature is None or gain <= 0:
            return self._leaf(y)
        
        # Split data
        mask = X[:, feature] <= threshold
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        
        return TreeNode(feature=feature, threshold=threshold, left=left, right=right)
    
    def _predict_one(self, x, node):
        """Predict single sample."""
        if node.is_leaf:
            return node.prediction
        
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X):
        """Predict class for X."""
        X = np.asarray(X, dtype=np.float32)
        return np.array([self._predict_one(x, self.root_) for x in X], dtype=int)
    
    def predict_proba(self, X):
        """Predict class probabilities (returns hard predictions for simplicity)."""
        preds = self.predict(X)
        n_classes = 2
        proba = np.zeros((len(preds), n_classes))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba
