def setup_dataset(dataset_id: str, drive_path: str, dataset_name: str):
    '''
    Mounts Google Drive, downloads the dataset from Kaggle, and moves it to Drive.
    Args:
        dataset_id (str): Kaggle dataset identifier
        drive_path (str): Google Drive path to store dataset
        dataset_name (str): Name for the dataset folder
    Returns:
        str: Path to dataset in Drive
    '''
    # Only mount if running in Colab
    import sys
    if 'google.colab' in sys.modules:
        from google.colab import drive as colab_drive
        colab_drive.mount('/content/drive')
    path = kagglehub.dataset_download(dataset_id)
    move_to_drive(path, drive_path, dataset_name)
    dest_path = os.path.join(drive_path, dataset_name)
    print(f"[INFO] Dataset available at: {dest_path}")
    return dest_path
import os
import shutil
import kagglehub
from google.colab import drive


def mount_drive():
    """
    Mount Google Drive in Colab.
    """
    drive.mount('/content/drive')


def download_dataset(dataset_id: str):
    """
    Download dataset using kagglehub.
    
    Args:
        dataset_id (str): Kaggle dataset identifier
    Returns:
        str: local download path
    """
    path = kagglehub.dataset_download(dataset_id)
    print(f"[INFO] Dataset downloaded to: {path}")
    return path


def move_to_drive(src_path: str, target_folder: str, dataset_name: str):
    """
    Move downloaded dataset to Google Drive.
    
    Args:
        src_path (str): Source dataset path
        target_folder (str): Destination folder in Drive
        dataset_name (str): Folder name for dataset
    """
    os.makedirs(target_folder, exist_ok=True)

    dest_path = os.path.join(target_folder, dataset_name)

    # If folder already exists, remove it to avoid errors
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    shutil.copytree(src_path, dest_path)

    print(f"[INFO] Dataset moved to: {dest_path}")


def main():
    # Kaggle dataset ID
    DATASET_ID = "vipoooool/new-plant-diseases-dataset"

    # Google Drive target path
    DRIVE_PATH = "/content/drive/MyDrive/ML/dataset"

    DATASET_NAME = "new-plant-diseases-dataset"

    # Step 1: Mount Drive
    mount_drive()

    # Step 2: Download dataset
    dataset_path = download_dataset(DATASET_ID)

    # Step 3: Move to Drive
    move_to_drive(dataset_path, DRIVE_PATH, DATASET_NAME)

    print("[DONE] Dataset setup complete.")


if __name__ == "__main__":
    main()
