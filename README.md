# TF-ICON Fork - Custom Modifications

This repository is a fork of the original [TF-ICON](https://github.com/original-repo/TF-ICON) with significant enhancements for advanced image composition and processing control.

## ✨ Key Features & Improvements

- **Enhanced Composition**: Improved background-foreground blending for more realistic outputs
- **Precise Positioning**: Dynamic control for foreground image placement
- **Smart Resizing**: Padding mechanism maintains aspect ratio while matching target dimensions
- **Flexible Configuration**: Expanded options for scaling factors and domain settings
- **Detailed Controls**: Enhanced argument parsing for seed control and image sources

## 🛠 Prerequisites

Before running the script, ensure you have:

- Python 3.x
- PyTorch
- tqdm
- subprocess
- shutil

## 📂 File Structure

```text
input/
│
├── cross_domain/          # Cross-domain dataset
│   ├── bg1.jpg            # Background image
│   ├── fg1.png            # Foreground image
│   ├── segmentation1.png  # Segmentation map
│   ├── position1.pkl      # Position data (pickle)
│   └── ...
│
├── same_domain/           # Same-domain dataset
│   ├── bg1.jpg            # Background image
│   ├── fg1.png            # Foreground image
│   ├── segmentation1.png  # Segmentation map
│   └── ...
└── ...
```

## 🚀 Usage

Run the batch processing script with:

```bash
python run_in_batches.py \
    --main_script "path/to/main_tf_icon.py" \
    --dataset_root "path/to/dataset" \
    --output_root "path/to/output" \
    --temp_input_root "path/to/temp_input" \
    --batch_size 4 \
    --cuda_num 5 \
    --domain "cross" \
    --dpm_steps 20 \
    --seed 3407
```

### 🔧 Command-line Arguments

| Argument            | Description                                      | Example Value                  |
|---------------------|--------------------------------------------------|--------------------------------|
| `--main_script`     | Path to main processing script                   | `scripts/main_tf_icon.py`      |
| `--dataset_root`    | Root directory of test cases                     | `./inputs/cross_domain`        |
| `--output_root`     | Directory for output results                     | `./outputs`                   |
| `--temp_input_root` | Temporary files directory                        | `./temp`                      |
| `--batch_size`      | Number of test cases per batch                   | `4`                           |
| `--cuda_num`        | GPU number to use                                | `0` (for cuda:0)              |
| `--domain`          | Processing domain (`cross` or `same`)            | `cross`                       |
| `--dpm_steps`       | Number of DPM steps                              | `20`                          |
| `--seed`            | Random seed for reproducibility                  | `1234`                        |

### 🧪 Example Command

```bash
python run_in_batches.py \
    --main_script "scripts/main_tf_icon.py" \
    --dataset_root "./inputs/cross_domain" \
    --output_root "./outputs" \
    --temp_input_root "./temp" \
    --batch_size 4 \
    --cuda_num 0 \
    --domain "cross" \
    --dpm_steps 20 \
    --seed 1234
```

## 🔄 Workflow

1. **Collect** test cases from `--dataset_root`
2. **Batch** them according to `--batch_size`
3. **Process** each batch:
   - Create temporary directory
   - Copy test cases
   - Execute main script
4. **Clean up** temporary files

## 💡 Notes

- Ensure all paths are correct and accessible
- The `--domain` argument affects model behavior (cross/same domain processing)
- Script handles errors gracefully and skips processed cases

## 📜 License

This project is licensed under the [MIT License](LICENSE).
