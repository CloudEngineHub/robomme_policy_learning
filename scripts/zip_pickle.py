import os
import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count


def _create_zip_batch_pkl(args):
    """
    Worker function to create a single zip file.
    
    Args:
        args: Tuple of (part_num, batch, target_path)
    """
    part_num, batch, target_path = args
    
    zip_name = f"part_{part_num}.zip"
    zip_path = target_path / zip_name
    
    print(f"Creating {zip_name} with files {len(batch)}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in batch:
            # Store with folder name (e.g., part1/0.pkl)
            arcname = f"{file_path.name}"
            zf.write(file_path, arcname)
    
    print(f"  Created {zip_name} ({len(batch)} files)")
    return zip_name


def zip_pickle_files(data_dir, target_dir, batch_size=50000, num_processes=None):
    """
    Zip .pkl files into part_{num}.zip archives and save to target directory using multiprocessing.
    
    Args:
        data_dir: Source directory containing {num}.pkl files
        target_dir: Target directory to save zip files
        batch_size: Number of files per zip (default: 50000)
    """

    data_path = Path(data_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Collect and sort all pickle files
    pkl_files = sorted(data_path.glob("*.pkl"))

    if not pkl_files:
        print("No .pkl files found to zip.")
        return

    # Prepare (part_num, batch, target_path) tuples
    batches = []
    for start in range(0, len(pkl_files), batch_size):
        part_num = start // batch_size
        batch = pkl_files[start : start + batch_size]
        batches.append((part_num, batch, target_path))

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    with Pool(processes=num_processes) as pool:
        results = pool.map(_create_zip_batch_pkl, batches)

    print(f"\nCompleted! Created {len(results)} part_*.zip files.")
    

# Example usage
if __name__ == "__main__":    
    # Example 1: Zip pickle files
    zip_pickle_files(
        data_dir="data/robomme_preprocessed_data/data", 
        target_dir="data/robomme_data_preprocessed_zipped/data", 
    )
    