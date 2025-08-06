# LoRA Processing Scripts

This directory contains batch scripts for processing LoRA models.

## Scripts

### `setup.bat`

This script sets up the Python environment for running the other scripts. It creates a virtual environment, activates it, and installs the required dependencies from the `requirements.txt` file in the parent directory.

**Usage:**

```bash
setup.bat
```

### `01-convert_and_resize_loras.bat`

This script converts and resizes LoRA models. It takes `.safetensors` files from the `input` folder, converts them from `ai-toolkit` to `sd-scripts` format using `convert_flux_lora.py`, resizes them using `networks\resize_lora.py`, and saves the output to the `output` folder. It also creates a log file for each processed file.

**Configuration:**

You can edit the following variables in the script to change its behavior:

- `new_rank`: The new rank for the resized LoRA model.
- `dyn_method`: The dynamic method to use for resizing.
- `dyn_param`: The dynamic parameter to use for resizing.
- `input_folder`: The folder where the input `.safetensors` files are located.
- `output_folder`: The folder where the resized `.safetensors` files will be saved.
- `temp_folder`: A temporary folder used for the conversion process.

**Usage:**

1. Place your `.safetensors` files in the `input` folder.
2. Run the script:

```bash
01-convert_and_resize_loras.bat
```

### `02-resize_loras.bat`

This script resizes LoRA models without converting them. It takes `.safetensors` files from the `input` folder, resizes them using `networks\resize_lora.py`, and saves the output to the `output` folder. It also creates a log file for each processed file.

**Configuration:**

You can edit the following variables in the script to change its behavior:

- `new_rank`: The new rank for the resized LoRA model.
- `dyn_method`: The dynamic method to use for resizing.
- `dyn_param`: The dynamic parameter to use for resizing.
- `input_folder`: The folder where the input `.safetensors` files are located.
- `output_folder`: The folder where the resized `.safetensors` files will be saved.

**Usage:**

1. Place your `.safetensors` files in the `input` folder.
2. Run the script:

```bash
02-resize_loras.bat
```

### `convert_flux_lora.py`

This Python script converts LoRA models between `ai-toolkit` and `sd-scripts` formats. It is used by the `01-convert_and_resize_loras.bat` script.

**Usage:**

```bash
python convert_flux_lora.py --src <source_format> --dst <destination_format> --src_path <source_path> --dst_path <destination_path>
```

**Arguments:**

- `--src`: The source format (`ai-toolkit` or `sd-scripts`).
- `--dst`: The destination format (`ai-toolkit` or `sd-scripts`).
- `--src_path`: The path to the source `.safetensors` file.
- `--dst_path`: The path to the destination `.safetensors` file.

```