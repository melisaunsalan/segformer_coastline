import os
import requests
import zipfile
import tarfile
import argparse

def download_dataset(dataset_name, save_dir="datasets", extract=True):
    """
    Downloads a dataset from a given URL and optionally extracts it if it's a zip or tar file.
    
    Args:
        dataset_name (str): The name of the dataset to download (e.g., "cifar10", "mnist").
        save_dir (str): The directory where the dataset will be saved.
        extract (bool): Whether to extract the dataset if it's a zip or tar file.
    """
    # Define a mapping for dataset names to URLs (add more datasets as needed)
    dataset_urls = {
        "snowed": "https://zenodo.org/records/8112715/files/SNOWED_v02.zip?download=1",
        "swed": "https://ukho-openmldata.s3.eu-west-2.amazonaws.com/SWED.zip"
    }

    if dataset_name not in dataset_urls:
        print(f"Dataset '{dataset_name}' not found in available datasets.")
        return

    # Get the dataset URL
    url = dataset_urls[dataset_name]
    dataset_name = os.path.basename(url)
    save_path = os.path.join(save_dir, dataset_name)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Downloading {dataset_name} from {url}...")

    # Download the dataset
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Dataset downloaded successfully: {save_path}")
        
        # Extract the dataset based on the file type
        if extract:
            if save_path.endswith(".zip"):
                extract_path = os.path.join(save_dir, dataset_name.replace(".zip", ""))
                with zipfile.ZipFile(save_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Dataset extracted to: {extract_path}")
            elif save_path.endswith((".tar", ".tar.gz", ".tgz")):
                extract_path = os.path.join(save_dir, dataset_name.replace(".tar.gz", "").replace(".tgz", "").replace(".tar", ""))
                with tarfile.open(save_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_path)
                print(f"Dataset extracted to: {extract_path}")
            else:
                print("Unsupported dataset type for extraction.")
    else:
        print(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract a dataset.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to download (e.g., 'snowed', 'swed')")
    parser.add_argument("--save_dir", type=str, default="datasets", help="The directory to save the dataset")
    parser.add_argument("--extract", action="store_true", help="Whether to extract the dataset after downloading")

    args = parser.parse_args()

    # Call the function with the arguments
    download_dataset(args.dataset_name, save_dir=args.save_dir, extract=args.extract)

if __name__ == "__main__":
    main()
