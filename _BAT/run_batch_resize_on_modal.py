# _BAT/run_batch_resize_on_modal.py
import modal
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# Your forked repository is now the source of truth.
GIT_REPO_URL = "https://github.com/tazztone/sd-scripts.git"
REPO_DIR = Path("/root/sd-scripts")

# Define the environment for the Modal app.
# It now clones your specific fork of the repository.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        f"git clone {GIT_REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && pip install -r requirements.txt",
        # Keeping specific dependencies from your setup for consistency
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 'numpy<2.0'",
    )
)

# Create a Modal App with the defined image
app = modal.App("tazztone-sd-scripts-processor", image=image)

# Create a persistent Volume for storing models.
# The structure (input, output, temp) remains the same.
volume = modal.Volume.from_name("lora-models-batch", create_if_missing=True)
BASE_DIR = "/models"
INPUT_DIR = Path(BASE_DIR, "input")
OUTPUT_DIR = Path(BASE_DIR, "output")
TEMP_DIR = Path(BASE_DIR, "temp")

def run_command(command: list[str]):
    """Helper function to run and stream output from a subprocess."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        cwd=REPO_DIR  # Run commands from the repo's root directory
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code {process.returncode}: {' '.join(command)}")

@app.function(
    gpu="any",
    volumes={BASE_DIR: volume},
    timeout=1800  # 30-minute timeout for batch processing
)
def process_loras_batch():
    """
    Runs the full convert-and-resize batch process using scripts from your repository.
    """
    # --- Configuration from your .bat file ---
    new_rank = 16
    dyn_method = "sv_fro"
    dyn_param = 0.94
    # -----------------------------------------

    print("--- Starting batch processing ---")
    
    # Ensure remote directories exist in the volume
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list(INPUT_DIR.glob("*.safetensors"))
    if not input_files:
        print("No .safetensors files found in /models/input. Aborting.")
        return

    print(f"Found {len(input_files)} models to process.")

    # Define paths to the scripts within the cloned repository
    convert_script_path = REPO_DIR / "_BAT" / "convert_flux_lora.py"
    resize_script_path = REPO_DIR / "networks" / "resize_lora.py"

    for src_path in input_files:
        filename_stem = src_path.stem
        extension = src_path.suffix
        print(f"\n--- Processing: {src_path.name} ---")

        # 1. Convert LoRA
        print("Step 1: Converting LoRA...")
        temp_filename = f"{filename_stem}-converted{extension}"
        temp_path = TEMP_DIR / temp_filename
        
        convert_command = [
            "python", str(convert_script_path),
            "--src", "ai-toolkit", "--dst", "sd-scripts",
            "--src_path", str(src_path), "--dst_path", str(temp_path),
        ]
        run_command(convert_command)

        # 2. Resize LoRA
        print(f"Step 2: Resizing '{temp_filename}'...")
        param_str = str(dyn_param).replace('.', '')
        new_filename = f"{filename_stem}-r{new_rank}-{dyn_method}-{param_str}{extension}"
        output_path = OUTPUT_DIR / new_filename

        resize_command = [
            "python", str(resize_script_path),
            "--model", str(temp_path),
            "--new_rank", str(new_rank),
            "--save_to", str(output_path),
            "--dynamic_method", dyn_method,
            "--dynamic_param", str(dyn_param),
            "--device", "cuda",
            "--save_precision", "fp16",
            "--verbose",
        ]
        run_command(resize_command)

        print(f"--- Finished processing {src_path.name}. Output saved to {output_path} ---")

    # 3. Cleanup
    print("\n--- Cleaning up temporary files ---")
    for temp_file in TEMP_DIR.glob("*"):
        temp_file.unlink()
    
    print("\nBatch processing complete. Forcing volume commit.")
    volume.commit()


@app.local_entrypoint()
def main():
    """
    Local entrypoint to trigger the remote batch processing function.
    """
    print("Calling remote function to process all LoRAs in the volume's '/input' directory.")
    process_loras_batch.remote()
