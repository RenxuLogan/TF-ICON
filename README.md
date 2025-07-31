# TF-ICON Fork - Custom Modifications

This repository is a fork of [TF-ICON](https://github.com/original-repo/TF-ICON) with several custom modifications aimed at improving image composition and adding more flexible control over image processing.

## Key Changes
- **Enhanced Background-Foreground Composition**: The blending logic between the background and foreground images has been improved for more realistic outputs.
- **Flexible Foreground Positioning**: Added dynamic control for foreground image positioning with options to specify precise areas for placement.
- **Padding and Resizing**: The padding mechanism ensures that images are resized to match the required target dimensions while maintaining their aspect ratio.
- **Configuration Flexibility**: More options added for controlling the scaling factor and domain for composition.
- **Detailed Argument Parsing**: New arguments for seed control, domain settings, and image sources.

## How to Use
To run the modified program, use the following command format:
```bash
python main.py --prompt "A scenic view of a mountain" --init-img "path/to/background.jpg" --ref-img "path/to/foreground.png" --seg "path/to/segmentation.png" --scale 2.5 --outdir "./output"
