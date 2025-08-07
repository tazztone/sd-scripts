# _BAT/modal_lora_converter.py
# Convert and optionally down-rank LoRA files on Modal

from pathlib import Path
import os, subprocess, logging
import modal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ───────────────────────────
# 1. Build Modal image
# ───────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")

    # ── system deps ──
    .apt_install("git", "libgl1")

    # ── PyPI wheels ──
    .pip_install(
        "torch==2.1.2","torchvision==0.16.2", extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        # long requirements list
        "numpy<2.0", "accelerate==0.30.0", "transformers==4.44.0",
        "diffusers[torch]==0.25.0", "ftfy==6.1.1",
        "opencv-python-headless==4.8.1.78", "einops==0.7.0",
        "pytorch-lightning==1.9.0", "bitsandbytes==0.44.0",
        "prodigyopt==1.0", "lion-pytorch==0.0.6", "tensorboard",
        "safetensors==0.4.2", "altair==4.2.2", "easygui==0.98.3",
        "toml==0.10.2", "voluptuous==0.13.1", "huggingface-hub==0.24.5",
        "imagesize==1.4.1", "rich==13.7.0",
    )
    .pip_install(
        "xformers==0.0.23.post1",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .env({"PYTHONPATH": "/sd-scripts"})

    # ── bring in repo twice:
    #    1) non-copy mount for runtime editing
    .add_local_dir(
        "c:/_coding/sd-scripts",
        "/sd-scripts",
        ignore=["venv/**", "__pycache__/**", ".git/**", "bitsandbytes_windows/**"],
    )
)

# ───────────────────────────
# 2. App, Volume, and Class
# ───────────────────────────
app = modal.App("lora-converter")
volume = modal.Volume.from_name("lora-conversion-volume", create_if_missing=True)

@app.cls(image=image, volumes={"/mnt/lora_volume": volume}, gpu="any")
class LoRAConverter:

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
        logger.info(f"Processing {src_path}…")

        filename = Path(src_path).stem
        ext = Path(src_path).suffix
        temp_fname = f"{filename}-converted{ext}"
        out_fname = f"{filename}-r{new_rank}-{dyn_method}-{str(dyn_param).replace('.','')}{ext}"

        # paths inside volume
        vol_in   = Path("/mnt/lora_volume") / src_path
        vol_tmp  = Path("/mnt/lora_volume") / temp_folder / temp_fname
        vol_out  = Path("/mnt/lora_volume") / output_folder / out_fname

        vol_tmp.parent.mkdir(parents=True, exist_ok=True)
        vol_out.parent.mkdir(parents=True, exist_ok=True)

        # — convert —
        convert_cmd = [
            "python", "/sd-scripts/_BAT/convert_flux_lora.py",
            "--src", "ai-toolkit",
            "--dst", "sd-scripts",
            "--src_path", str(vol_in),
            "--dst_path", str(vol_tmp),
        ]
        self._run(convert_cmd, "Conversion")

        # — resize —
        resize_cmd = [
            "python", "/sd-scripts/networks/resize_lora.py",
            "--model", str(vol_tmp),
            "--new_rank", str(new_rank),
            "--save_to", str(vol_out),
            "--dynamic_method", dyn_method,
            "--dynamic_param", str(dyn_param),
            "--device", "cuda",
            "--save_precision", "fp16",
            "--verbose",
        ]
        self._run(resize_cmd, "Resize")

        # cleanup
        vol_tmp.unlink(missing_ok=True)
        logger.info(f"Finished {src_path} → {vol_out}")

    # helper that logs stdout/stderr and raises on non-zero
    def _run(self, cmd, title):
        logger.info(f"{title}: {' '.join(cmd)}")
        res = subprocess.run(cmd, text=True, capture_output=True)
        if res.stdout:
            logger.info(res.stdout)
        if res.stderr:
            logger.error(res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"{title} failed")

# ───────────────────────────
# 3. Local entrypoint
# ───────────────────────────
@app.local_entrypoint()
def main(
    new_rank: int = 32,
    dyn_method: str = "sv_fro",
    dyn_param: float = 0.94,
    input_folder: str = "_BAT/input",
    output_folder: str = "_BAT/output",
    temp_folder: str = "_BAT/temp",
):
    # copy local .safetensors into volume
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    for f in Path(input_folder).glob("*.safetensors"):
        target = Path("/mnt/lora_volume") / input_folder / f.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(f.read_bytes())
            logger.info(f"Copied {f} → {target}")

    files = list((Path("/mnt/lora_volume") / input_folder).glob("*.safetensors"))
    if not files:
        logger.warning("No .safetensors found.")
        return

    converter = LoRAConverter()
    for f in files:
        converter.convert_and_resize_lora.remote(
            src_path=str(f.relative_to("/mnt/lora_volume").as_posix()),
            new_rank=new_rank,
            dyn_method=dyn_method,
            dyn_param=dyn_param,
            output_folder=output_folder,
            temp_folder=temp_folder,
        )
    logger.info("Submitted all jobs.")
