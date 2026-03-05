import os
import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count


def _create_zip_batch(args):
    """
    Worker function to create a single zip file.
    
    Args:
        args: Tuple of (batch, part_num, target_path)
    """
    task, batch, target_path = args
    
    zip_name = f"{task}.zip"
    zip_path = target_path / zip_name
    
    print(f"Creating {zip_name} with files {len(batch)}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in batch:
            # Store with folder name (e.g., part1/0.pkl)
            arcname = f"{file_path.name}"
            zf.write(file_path, arcname)
    
    print(f"  Created {zip_name} ({len(batch)} files)")
    return zip_name



TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]


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
    pkl_files = pkl_files[:50000]

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
    
        

def zip_files_in_batches(data_dir, target_dir, batch_size=5000, num_processes=None):
    """
    Zip .pkl files into batches and save to target directory using multiprocessing.
    
    Args:
        data_dir: Source directory containing {num}.pkl files
        target_dir: Target directory to save zip files
        batch_size: Number of files per zip (default: 5000)
    """
    data_path = Path(data_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    pkl_file_dic = {task: [] for task in TASK_NAME_LIST}
    for f in data_path.glob("*.png"):
        task = f.stem.split("_")[1]
        if task in pkl_file_dic:
            pkl_file_dic[task].append(f)
        
    # Prepare batches
    batches = []
    print(f"total number of files: {len(pkl_file_dic)}")
    
    for task in TASK_NAME_LIST:
        batch = pkl_file_dic[task]
        batches.append((task, batch, target_path))
        
    with Pool(processes=10) as pool:
        results = pool.map(_create_zip_batch, batches)
    
    print(f"\nCompleted! Created {len(results)} zip files.")
    
    

def zip_files_in_batches_mp4(data_dir, target_dir, batch_size=5000, num_processes=None):
    """
    Zip .pkl files into batches and save to target directory using multiprocessing.
    
    Args:
        data_dir: Source directory containing {num}.pkl files
        target_dir: Target directory to save zip files
        batch_size: Number of files per zip (default: 5000)
    """
    data_path = Path(data_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    mp4_files = []
    for f in data_path.glob("*.mp4"):
        mp4_files.append(f)
                
        
    with Pool(processes=10) as pool:
        results = pool.map(_create_zip_batch, [(mp4_files, "all_videos", target_path)])
    
    print(f"\nCompleted! Created {len(results)} zip files.")


def _extract_flat_worker(args):
    """
    Worker function to extract a single zip file (flat structure).
    
    Args:
        args: Tuple of (zip_file, output_dir)
    """
    zip_file, output_dir = args
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_file.name} to {output_dir} (flat structure)...")
    
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for member in zf.namelist():
            # Extract only the filename, ignore directory structure
            filename = os.path.basename(member)
            if filename:  # Skip if it's just a directory entry
                source = zf.open(member)
                target = open(output_path / filename, "wb")
                with source, target:
                    target.write(source.read())
    
    print(f"  Extracted {zip_file.name}")
    return zip_file.name


def _extract_with_folders_worker(args):
    """
    Worker function to extract a single zip file (with folder structure).
    
    Args:
        args: Tuple of (zip_file, output_dir)
    """
    zip_file, output_dir = args
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_file.name} to {output_dir} (with folders)...")
    
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(output_path)
    
    print(f"  Extracted {zip_file.name}")
    return zip_file.name


def extract_flat(zip_file, output_dir):
    """
    Extract zip file with all .pkl files directly in output_dir (no subfolder).
    
    Args:
        zip_file: Path to the zip file
        output_dir: Directory to extract files to
    """
    _extract_flat_worker((Path(zip_file), output_dir))


def extract_with_folders(zip_file, output_dir):
    """
    Extract zip file maintaining the folder structure (e.g., part1/, part2/).
    
    Args:
        zip_file: Path to the zip file
        output_dir: Directory to extract files to
    """
    _extract_with_folders_worker((Path(zip_file), output_dir))


def extract_all_zips(zip_dir, output_dir, flat=True):
    """
    Extract all part*.zip files from zip_dir to output_dir using multiprocessing.
    
    Args:
        zip_dir: Directory containing part*.zip files
        output_dir: Directory to extract files to
        flat: If True, extract flat; if False, maintain folder structure
    """
    zip_path = Path(zip_dir)
    zip_files = sorted(zip_path.glob("part*.zip"))
    
    if not zip_files:
        print("No part*.zip files found!")
        return
    
    # Prepare arguments for worker functions
    args_list = [(zip_file, output_dir) for zip_file in zip_files]
    
    worker_func = _extract_flat_worker if flat else _extract_with_folders_worker
    
    print(f"Extracting {len(zip_files)} zip files ...\n")
    
    with Pool(processes=10) as pool:
        results = pool.map(worker_func, args_list)
    
    print(f"\nCompleted! Extracted {len(results)} zip files.")

# Example usage
if __name__ == "__main__":
    # # # Example 1: Zip files
    # zip_files_in_batches_mp4(
    #     data_dir="/home/daiyp/openpi/examples/history_bench_sim/qwen3_vl/HistoryBenchMemER/final_data_memER/images", 
    #     target_dir="/home/daiyp/openpi/examples/history_bench_sim/qwen3_vl/HistoryBenchMemER/final_data_memER", 
    # )
    
    
    # Example 1: Zip pickle files
    zip_pickle_files(
        data_dir="data/robomme_preprocessed_data/data", 
        target_dir="data/robomme_data_preprocessed_zipped/data", 
    )
    
    
    # # Example 2: Extract flat (all .pkl files in same folder)
    # extract_all_zips(
    #     zip_dir="data/robomme_preprocessed_data/data", 
    #     output_dir="data/robomme_data_preprocessed_zipped/data", 
    #     flat=True
    # )