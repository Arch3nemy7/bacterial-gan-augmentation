import os
import random
import glob

def balance_directory(base_path):
    subsets = ['train', 'val', 'test']
    classes = ['gram_negative', 'gram_positive']
    
    for subset in subsets:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            print(f"Skipping {subset}, not found.")
            continue
            
        print(f"Processing {subset}...")
        
        counts = {}
        files_dict = {}
        
        for cls in classes:
            cls_path = os.path.join(subset_path, cls)
            # Use glob to find files, assuming they are files not directories
            files = [f for f in glob.glob(os.path.join(cls_path, '*')) if os.path.isfile(f)]
            counts[cls] = len(files)
            files_dict[cls] = files
            print(f"  {cls}: {len(files)} files")
            
        if not counts:
             print(f"  No classes found in {subset}.")
             continue

        min_count = min(counts.values())
        target_count = min_count
        
        for cls in classes:
            if counts[cls] > target_count:
                diff = counts[cls] - target_count
                print(f"  Trimming {cls} by {diff} files to match {target_count}...")
                
                # Randomly shuffle and pick files to remove
                # We use random.sample to pick 'diff' items to remove
                files_to_remove = random.sample(files_dict[cls], diff)
                
                for f in files_to_remove:
                    os.remove(f)
                
                print(f"  Removed {len(files_to_remove)} files from {cls}.")
            else:
                print(f"  {cls} is already at target count ({target_count}).")

if __name__ == "__main__":
    random.seed(42)
    base_dir = "data/02_processed"
    # Ensure we are running from the project root or adjust path
    if not os.path.exists(base_dir):
        # Try absolute path if relative fails, or check if we are in scripts dir
        if os.path.exists(os.path.join("..", base_dir)):
             base_dir = os.path.join("..", base_dir)
        else:
             print(f"Error: Could not find {base_dir}")
             exit(1)
             
    balance_directory(base_dir)
