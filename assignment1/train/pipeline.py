import pandas as pd
import matplotlib.pyplot as plt

# --- Scikit-Learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load the cleaned dataset
    try:
        df = pd.read_csv('cleaned.csv')
    except FileNotFoundError:
        print("Error: 'cleaned.csv' not found. Please ensure it is in the same directory.")
        return

    # Extract features (all except last) and labels (last column)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # 2. Analyze class balance
    print("--- Class Balance ---")
    class_counts = y.value_counts()
    print(class_counts.to_string())
    
    if len(class_counts) > 0 and class_counts.min() < (0.5 * class_counts.max()):
        print("\nConclusion: The dataset appears to be imbalanced.\n")
    else:
        print("\nConclusion: The dataset is relatively balanced.\n")

    # 3. Train-test split (80/20 ratio) using scikit-learn
    # Note: 'stratify=y' ensures the 80/20 split maintains the same class proportions!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=94, 
        stratify=y
    )

    print(f"--- Train-Test Split ---")
    print(f"Ratio: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)\n")

    # 4 & 5. Evaluate test accuracy and experiment with k=1 to 10
    print("--- Experimenting with k values ---")
    k_values = list(range(1, 11))
    accuracies = []

    for k in k_values:
        # Initialize the scikit-learn KNN model
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Accuracy for k={k}: {acc*100:.2f}%")

    # Plot Accuracy vs. k and save to PDF
    plt.figure(figsize=(8, 5))
    # Changed color to green just to easily distinguish this plot from your manual one
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='g') 
    plt.title('Scikit-Learn KNN Accuracy vs. k (Wheat Dataset)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Test Accuracy')
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Exporting to the requested PDF name
    pdf_filename = 'scikit_learn_KNN.pdf'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"\nPlot successfully saved to {pdf_filename}")

if __name__ == "__main__":
    main()
