import pandas as pd

def clean_data(input_file='data.csv', output_file='cleaned.csv'):
    print(f"Reading raw data from '{input_file}'...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please ensure it is in the same directory.")
        return

    original_shape = df.shape

    # 1. Force a strict column cutoff
    # We know the dataset should only have exactly 8 columns. 
    # This aggressively slices off the trailing comma ghost columns (Unnamed: 8, Unnamed: 9, etc.)
    df = df.iloc[:, :8]

    # 2. Force numeric types
    # This turns any strings, spaces (' '), or unparseable text into NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # 3. Drop corrupted or incomplete rows
    # Now this will only drop genuinely broken rows (like the lone '1' at the bottom)
    df = df.dropna()

    # 4. Clean up the Class column
    # Classes should be integers (1, 2, 3) rather than floats (1.0, 2.0, 3.0)
    class_col_name = df.columns[-1]
    df[class_col_name] = df[class_col_name].astype(int)

    # 5. Export to the new CSV
    df.to_csv(output_file, index=False)
    
    print("--- Cleaning Complete ---")
    print(f"Original shape: {original_shape[0]} rows, {original_shape[1]} columns")
    print(f"Cleaned shape:  {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Successfully exported clean dataset to '{output_file}'!")

if __name__ == "__main__":
    clean_data()
