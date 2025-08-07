from io import BytesIO
from pathlib import Path
import modal
import os
import subprocess
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the Modal image with necessary dependencies
# We'll use a base image and then install the requirements from requirements.txt
# Note: Some packages in requirements.txt might need specific handling or might not be compatible with Modal's environment.
# We'll start with a general approach and refine if errors occur.

# Read requirements.txt to get the list of packages locally
# This runs on your local machine before the Modal image is built
local_requirements_path = Path("requirements.txt") # Assuming requirements.txt is in the project root
if not local_requirements_path.exists():
    # If the script is run from _BAT, then requirements.txt is one level up
    local_requirements_path = Path(__file__).parent.parent / "requirements.txt"

if not local_requirements_path.exists():
    raise FileNotFoundError(f"requirements.txt not found at {local_requirements_path}")

with open(local_requirements_path, "r") as f:
    pip_packages = [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-e")]

# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git") # Assuming git might be needed for some packages or operations
    .pip_install(
        "torch==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "xformers==0.0.23.post1",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(pip_packages)
    .add_local_dir("c:/_coding/sd-scripts", "/sd-scripts", ignore=["venv/**", "__pycache__/**", ".git/**", "bitsandbytes_windows/**", "library.egg-info/**"]) # Add the entire project directory, excluding venv and pycache
)

app = modal.App("lora-converter")

# Define a Modal Volume for persistent storage of input/output/temp files
# This allows files to persist across Modal runs
volume = modal.Volume.from_name("lora-conversion-volume", create_if_missing=True)

@app.cls(image=image, volumes={"/mnt/lora_volume": volume}, gpu="any") # Using "any" for GPU, adjust as needed
class LoRAConverter:
    def __enter__(self):
        # Ensure the necessary scripts are available in the Modal environment
        # We'll copy them from the local project to the Modal volume if they don't exist
        # This is a simplified approach; for a real application, you might want to
        # include them directly in the image or use modal.Mount
        
        # Create directories within the volume if they don't exist
        os.makedirs("/mnt/lora_volume/_BAT/input", exist_ok=True)
        os.makedirs("/mnt/lora_volume/_BAT/output", exist_ok=True)
        os.makedirs("/mnt/lora_volume/_BAT/temp", exist_ok=True)

        # Copy the Python scripts to the volume if they are not already there
        # This is a placeholder. In a real scenario, you'd use modal.Mount
        # to make your local files available in the container.
        # For now, we'll assume the scripts are part of the image or manually copied.
        pass

    @modal.method()
    def convert_and_resize_lora(
        self,
        src_path: str,
        new_rank: int,
        dyn_method: str,
        dyn_param: float,
        output_folder: str,
        temp_folder: str,
    ):
        logger.info(f"Processing {src_path}...")

        filename = Path(src_path).stem
        extension = Path(src_path).suffix
        new_filename = f"{filename}-r{new_rank}-{dyn_method}-{str(dyn_param).replace('.', '')}{extension}"
        log_filename = f"{filename}-r{dyn_method}-{str(dyn_param).replace('.', '')}.txt"
        temp_filename = f"{filename}-converted{extension}"

        # Define paths within the Modal volume
        modal_src_path = Path("/mnt/lora_volume") / src_path
        modal_output_folder = Path("/mnt/lora_volume") / output_folder
        modal_temp_folder = Path("/mnt/lora_volume") / temp_folder
        
        modal_temp_file_path = modal_temp_folder / temp_filename
        modal_dst_file_path = modal_output_folder / new_filename
        modal_log_file_path = modal_output_folder / log_filename

        # Ensure output and temp directories exist within the volume
        os.makedirs(modal_output_folder, exist_ok=True)
        os.makedirs(modal_temp_folder, exist_ok=True)

        # Step 1: Convert LoRA
        convert_cmd = [
            "python",
            "/sd-scripts/_BAT/convert_flux_lora.py", # Adjust path as necessary within the container
            "--src", "ai-toolkit",
            "--dst", "sd-scripts",
            "--src_path", str(modal_src_path),
            "--dst_path", str(modal_temp_file_path),
        ]
        logger.info(f"Running conversion: {' '.join(convert_cmd)}")
        try:
            convert_result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            logger.info(convert_result.stdout)
            if convert_result.stderr:
                logger.error(convert_result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise

        # Step 2: Resize LoRA
        resize_cmd = [
            "python",
            "/sd-scripts/networks/resize_lora.py", # Adjust path as necessary within the container
            "--model", str(modal_temp_file_path),
            "--new_rank", str(new_rank),
            "--save_to", str(modal_dst_file_path),
            "--dynamic_method", dyn_method,
            "--dynamic_param", str(dyn_param),
            "--device", "cuda", # Assuming GPU is available and configured
            "--save_precision", "fp16",
            "--verbose",
        ]
        logger.info(f"Running resize: {' '.join(resize_cmd)}")
        try:
            resize_result = subprocess.run(resize_cmd, capture_output=True, text=True, check=True)
            logger.info(resize_result.stdout)
            if resize_result.stderr:
                logger.error(resize_result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"Resizing failed: {e.stderr}")
            raise

        # Clean up temporary file
        os.remove(modal_temp_file_path)
        logger.info(f"Cleaned up {modal_temp_file_path}")

        logger.info(f"Processed {src_path}, output saved to {modal_dst_file_path}")
        
        # Log to a file as well, similar to the .bat script
        with open(modal_log_file_path, "w") as f:
            log_content = f"--- Converting {src_path} ---\n"
            log_content += f"Conversion command: {' '.join(convert_cmd)}\n"
            log_content += convert_result.stdout
            if convert_result.stderr:
                log_content += f"Errors: {convert_result.stderr}\n"
            log_content += f"\n--- Resizing {temp_filename} ---\n"
            log_content += f"Resize command: {' '.join(resize_cmd)}\n"
            log_content += resize_result.stdout
            if resize_result.stderr:
                log_content += f"Errors: {resize_result.stderr}\n"
            log_content += f"Processed {src_path}, output saved to {modal_dst_file_path}\n"
            f.write(log_content)


@app.local_entrypoint()
def main(
    new_rank: int = 32,
    dyn_method: str = "sv_fro",
    dyn_param: float = 0.94,
    input_folder: str = "_BAT/input",
    output_folder: str = "_BAT/output",
    temp_folder: str = "_BAT/temp",
):
    # Ensure local input, output, and temp folders exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    # Copy local input files to the Modal volume
    # This is crucial for Modal to access the files
    local_input_path = Path(input_folder)
    for file_path in local_input_path.glob("*.safetensors"):
        logger.info(f"Copying {file_path} to Modal volume...")
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Write to the Modal volume
        # The path inside the volume should mirror the local path for consistency
        modal_volume_path = Path("/mnt/lora_volume") / file_path
        modal_volume_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist in volume
        with open(modal_volume_path, "wb") as f:
            f.write(content)
        logger.info(f"Copied {file_path} to {modal_volume_path}")

    # Get a list of .safetensors files in the input folder within the Modal volume
    # This assumes the files have been copied to the volume
    modal_input_folder = Path("/mnt/lora_volume") / input_folder
    input_files = list(modal_input_folder.glob("*.safetensors"))

    if not input_files:
        logger.warning(f"No .safetensors files found in {modal_input_folder}. Please place your LoRA files in this directory.")
        return

    converter = LoRAConverter()
    for file_path in input_files:
        # Call the remote Modal method
        converter.convert_and_resize_lora.remote(
            src_path=str(file_path.relative_to("/mnt/lora_volume")), # Pass relative path for the remote method
            new_rank=new_rank,
            dyn_method=dyn_method,
            dyn_param=dyn_param,
            output_folder=output_folder,
            temp_folder=temp_folder,
        )

    logger.info("All LoRAs have been processed. Results are in the output folder within the Modal volume.")

    # After processing, you might want to copy results back to local machine
    # This part is not implemented here, but would involve reading from the volume
    # and writing to the local filesystem.
    
    # Clean up temporary files in the Modal volume
    modal_temp_folder = Path("/mnt/lora_volume") / temp_folder
    for temp_file in modal_temp_folder.glob("*.safetensors"):
        os.remove(temp_file)
        logger.info(f"Cleaned up temporary file: {temp_file}")
    
    # Remove the temporary directory
    if os.path.exists(modal_temp_folder) and not os.listdir(modal_temp_folder):
        os.rmdir(modal_temp_folder)
        logger.info(f"Removed temporary directory: {modal_temp_folder}")

    logger.info("Modal LoRA conversion and resizing complete.")
