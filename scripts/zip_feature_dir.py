import os
import zipfile
from pathlib import Path

def zip_episode_folders(base_path, target_directory):
    """
    Zips each episode folder in the specified directory.
    
    Args:
        base_path: Path to the directory containing episode folders (source)
        target_directory: Path where zipped episode files will be written
    """
    base_path = Path(base_path)
    target_directory = Path(target_directory)
    
    if not base_path.exists():
        print(f"Error: Path '{base_path}' does not exist")
        return
    
    # Ensure target directory exists
    target_directory.mkdir(parents=True, exist_ok=True)

    # Get all episode folders from the source directory
    episode_folders = [f for f in base_path.iterdir() 
                      if f.is_dir() and f.name.startswith('episode_')]
    
    if not episode_folders:
        print("No episode folders found")
        return
    
    print(f"Found {len(episode_folders)} episode folders")
    
    for folder in sorted(episode_folders[:2]):
        # Write zip files into the target directory while zipping contents
        # from the source directory.
        zip_name = target_directory / f"{folder.name}.zip"
        
        print(f"Zipping {folder.name}...")
        
        try:
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the folder and add all files
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        file_path = Path(root) / file
                        # Archive name relative to the source base directory
                        arcname = file_path.relative_to(base_path)
                        zipf.write(file_path, arcname)
            
            print(f"✓ Created {zip_name.name}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    # Change this path to match your directory structure
    base_directory = "data/robomme_preprocessed_data/features"
    target_directory = "data/robomme_data_preprocessed_zipped/features"

    zip_episode_folders(base_directory, target_directory)