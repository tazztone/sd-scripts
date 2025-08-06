@echo off
cd ..
call .\venv\Scripts\activate.bat
setlocal enabledelayedexpansion

REM Configuration
set "new_rank=16"
set "dyn_method=sv_fro"
set "dyn_param=0.94"
set "input_folder=_BAT\input"
set "output_folder=_BAT\output"

REM Ensure output directory exists
if not exist "%output_folder%" mkdir "%output_folder%"

REM Process each .safetensors file in the input folder
for %%f in ("%input_folder%\*.safetensors") do (
    set "filename=%%~nf"
    set "extension=%%~xf"
    set "new_filename=!filename!-r%new_rank%-%dyn_method%-%dyn_param:.=%!extension!"
    echo Processing %%f...
    
    python networks\resize_lora.py --model "%%f" --network_dim %original_rank% --new_rank %new_rank% --save_to "%output_folder%\!new_filename!" --dynamic_method %dyn_method% --dynamic_param %dyn_param% --device cuda --save_precision fp16 --verbose
)

echo.
echo All LoRAs have been resized.
pause