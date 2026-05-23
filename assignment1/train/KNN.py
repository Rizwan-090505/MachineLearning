import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(x, y):
    """Calculates the Euclidean distance between two numpy arrays."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    difference = x - y
    return np.sqrt(np.sum(difference ** 2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy(dtype=float) if isinstance(X_train, pd.DataFrame) else np.array(X_train, dtype=float)
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else np.array(y_train)

    def predict(self, point):
        point = np.array(point, dtype=float)
        distances = []
        
        for i in range(len(self.X_train)):
            dist = euclidean_distance(point, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        
        votemap = {}
        for label in k_nearest_labels:
            votemap[label] = votemap.get(label, 0) + 1
            
        return max(votemap, key=votemap.get)

    def evaluate(self, X_test, y_test):
        X_test_np = X_test.to_numpy(dtype=float) if isinstance(X_test, pd.DataFrame) else np.array(X_test, dtype=float)
        y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.array(y_test)
        
        correct = 0
        for i in range(len(X_test_np)):
            prediction = self.predict(X_test_np[i])
            if prediction == y_test_np[i]:
                correct += 1
                
        accuracy = correct / len(X_test_np)
        return accuracy

def main():
    # 1. Load the dataset
    try:
        df = pd.read_csv('cleaned.csv')
    except FileNotFoundError:
        print("Error: 'data.csv' not found. Please ensure it is in the same directory.")
        return

    # --- FOOLPROOF DATA CLEANING ---
    # 1. Drop any ghost columns caused by trailing commas (,,)
    df = df.dropna(axis=1, how='all')
    
    # 2. Force everything to numbers. Turns spaces (' ') into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 3. Drop any rows that now contain NaN to ensure perfect, clean data
    df = df.dropna()
    # -------------------------------

    # Extract features (all except last) and labels (last column)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # 3. Analyze class balance
    print("--- Class Balance ---")
    class_counts = y.value_counts()
    print(class_counts.to_string())
    
    if len(class_counts) > 0 and class_counts.min() < (0.5 * class_counts.max()):
        print("\nConclusion: The dataset appears to be imbalanced.\n")
    else:
        print("\nConclusion: The dataset is relatively balanced.\n")

    # 2. Train-test split (80/20 ratio)
    np.random.seed(94) 
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    print(f"--- Train-Test Split ---")
    print(f"Ratio: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)\n")

    # 4 & 5. Evaluate test accuracy and experiment with k=1 to 10
    print("--- Experimenting with k values ---")
    k_values = list(range(1, 11))
    accuracies = []

    for k in k_values:
        model = KNN(k=k)
        model.fit(X_train, y_train)
        acc = model.evaluate(X_test, y_test)
        accuracies.append(acc)
        print(f"Accuracy for k={k}: {acc*100:.2f}%")

    # Plot Accuracy vs. k and save to PDF
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs. k (Wheat Dataset)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Test Accuracy')
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    pdf_filename = 'KNN_Accuracy_Report.pdf'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"\nPlot successfully saved to {pdf_filename}")

if __name__ == "__main__":
    main()
