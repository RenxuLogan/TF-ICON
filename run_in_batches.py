import os
import subprocess
import shutil
from tqdm import tqdm
from itertools import islice
import argparse

# ==============================================================================
# 1. Configuration Parameters (Modified for user)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Batch processing for TF-ICON based image composition.")
    parser.add_argument("--main_script", type=str, required=True, help="Path to the main script to run")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for output results")
    parser.add_argument("--temp_input_root", type=str, required=True, help="Root directory for temporary input files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--cuda_num", type=int, default=5, help="GPU number to use")
    parser.add_argument("--domain", type=str, choices=["cross", "same"], required=True, help="Domain type ('cross' or 'same')")
    parser.add_argument("--dpm_steps", type=int, default=20, help="Number of DPM steps")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    # Parse user arguments
    opt = parse_args()

    # Ensure main script exists
    if not os.path.exists(opt.main_script):
        print(f"[Error] Main script not found: {opt.main_script}")
        return

    # 1. Collect all test case directories
    all_case_dirs = []
    try:
        case_names = sorted(os.listdir(opt.dataset_root))
    except FileNotFoundError:
        print(f"[Error] Dataset root directory not found: {opt.dataset_root}")
        return
        
    for case_name in case_names:
        case_path = os.path.join(opt.dataset_root, case_name)
        if not os.path.isdir(case_path):
            continue
        
        final_output_path = os.path.join(opt.output_root, opt.domain, case_name.replace(" ", "_"), "result.png")
        if os.path.exists(final_output_path):
            continue
            
        all_case_dirs.append(case_path)
            
    if not all_case_dirs:
        print("No cases to process, or all cases have already been processed.")
        return

    print(f"Found {len(all_case_dirs)} test cases to process. Will process in batches of {opt.batch_size}...")

    # 2. Split cases into batches
    it = iter(all_case_dirs)
    batches = list(iter(lambda: tuple(islice(it, opt.batch_size)), ()))

    for i, batch in enumerate(tqdm(batches, desc=f"Processing batches, GPU={opt.cuda_num}")):
        
        print(f"\n--- Processing batch {i + 1} ---")
        
        # 3. Prepare temporary input directory
        temp_domain_dir = os.path.join(opt.temp_input_root, opt.domain)
        
        if os.path.exists(temp_domain_dir):
            shutil.rmtree(temp_domain_dir)
        os.makedirs(temp_domain_dir)
        
        print(f"   -> Preparing temporary folder: {temp_domain_dir}")

        # 4. Copy case folders to temporary directory
        for case_path in batch:
            case_name = os.path.basename(case_path)
            dest_path = os.path.join(temp_domain_dir, case_name)
            try:
                shutil.copytree(case_path, dest_path)
                print(f"    - Copied: {case_name}")
            except Exception as e:
                print(f"    - [Error] Failed to copy {case_name}: {e}")
                continue

        # 5. Build command to execute main program
        command = [
            "python", opt.main_script,
            "--root", opt.temp_input_root,
            "--outdir", opt.output_root,
            "--domain", opt.domain,
            "--gpu", f"cuda:{opt.cuda_num}",
            "--dpm_steps", str(opt.dpm_steps),
            "--seed", str(opt.seed)
        ]

        print("   -> Running the main script for current batch...")
        try:
            subprocess.run(command, check=True)
            print(f"   -> Batch {i + 1} processed successfully!")
        except subprocess.CalledProcessError:
            print(f"   -> [Error] Batch {i + 1} failed!")

    # 6. Clean up temporary folder
    if os.path.exists(opt.temp_input_root):
        shutil.rmtree(opt.temp_input_root)
    print("\nAll batches processed!")

if __name__ == '__main__':
    main()
