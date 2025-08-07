import modal
import subprocess
import sys
from pathlib import Path

# Define the environment for the Modal app
# This clones the repository and installs dependencies.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/kohya-ss/sd-scripts.git /root/sd-scripts",
        "cd /root/sd-scripts && pip install -r requirements.txt",
        # Note: numpy<2.0 is specified in your setup.bat and might be important
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 'numpy<2.0'",
    )
)

# Create a Modal App with the defined image
app = modal.App("kohya-ss-batch-processor", image=image)

# Create a persistent Volume for storing models and scripts
# This volume is structured with input, output, and temp directories
volume = modal.Volume.from_name("lora-models-batch", create_if_missing=True)
BASE_DIR = "/models"
INPUT_DIR = Path(BASE_DIR, "input")
OUTPUT_DIR = Path(BASE_DIR, "output")
TEMP_DIR = Path(BASE_DIR, "temp")

# Mount your local convert_flux_lora.py file into the container
# Make sure it's in the same directory as this launcher script.
mounts = [
    modal.Mount.from_local_file(
        "convert_flux_lora.py", remote_path="/root/convert_flux_lora.py"
    )
]

def run_command(command: list[str]):
    """Helper function to run and stream output from a subprocess."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
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
    mounts=mounts,
    timeout=1800  # Set a 30-minute timeout for batch processing
)
def process_loras_batch():
    """
    Runs the full convert-and-resize batch process on all models
    in the volume's /input directory.
    """
    # --- Configuration from your .bat file ---
    new_rank = 16
    dyn_method = "sv_fro"
    dyn_param = 0.94
    # -----------------------------------------

    print("--- Starting batch processing ---")
    
    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list(INPUT_DIR.glob("*.safetensors"))
    if not input_files:
        print("No .safetensors files found in /models/input. Aborting.")
        return

    print(f"Found {len(input_files)} models to process.")

    for src_path in input_files:
        filename_stem = src_path.stem
        extension = src_path.suffix

        print(f"\n--- Processing: {src_path.name} ---")

        # 1. Convert LoRA
        print("Step 1: Converting LoRA from ai-toolkit to sd-scripts format...")
        temp_filename = f"{filename_stem}-converted{extension}"
        temp_path = TEMP_DIR / temp_filename
        
        convert_command = [
            "python", "/root/convert_flux_lora.py",
            "--src", "ai-toolkit",
            "--dst", "sd-scripts",
            "--src_path", str(src_path),
            "--dst_path", str(temp_path),
        ]
        run_command(convert_command)

        # 2. Resize LoRA
        print(f"Step 2: Resizing '{temp_filename}'...")
        # Recreate the output filename from your .bat script
        param_str = str(dyn_param).replace('.', '')
        new_filename = f"{filename_stem}-r{new_rank}-{dyn_method}-{param_str}{extension}"
        output_path = OUTPUT_DIR / new_filename

        resize_command = [
            "python", "/root/sd-scripts/networks/resize_lora.py",
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

