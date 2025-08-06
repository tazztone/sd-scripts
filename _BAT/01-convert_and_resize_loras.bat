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
set "temp_folder=_BAT\temp"

REM Ensure output and temp directories exist
if not exist "%output_folder%" mkdir "%output_folder%"
if not exist "%temp_folder%" mkdir "%temp_folder%"

REM Process each .safetensors file in the input folder
for %%f in ("%input_folder%\*.safetensors") do (
    set "filename=%%~nf"
    set "extension=%%~xf"
    set "new_filename=!filename!-r%new_rank%-%dyn_method%-%dyn_param:.=%%!extension!"
    set "log_filename=!filename!-r%new_rank%-%dyn_method%-%dyn_param:.=%%.txt"
    set "temp_filename=!filename!-converted!extension!"

    (
        echo --- Converting %%f ---
        python _BAT\convert_flux_lora.py --src "ai-toolkit" --dst "sd-scripts" --src_path "%%f" --dst_path "%temp_folder%\!temp_filename!"
        
        echo.
        echo --- Resizing !temp_filename! ---
        python networks\resize_lora.py --model "%temp_folder%\!temp_filename!" --new_rank %new_rank% --save_to "%output_folder%\!new_filename!" --dynamic_method %dyn_method% --dynamic_param %dyn_param% --device cuda --save_precision fp16 --verbose
    ) > "%output_folder%\!log_filename!" 2>&1

    echo Processed %%f, log saved to %output_folder%\!log_filename!
)

echo.
echo --- Cleaning up temporary files ---
del "%temp_folder%\*.safetensors"
rmdir "%temp_folder%"

echo.
echo All LoRAs have been resized.
pause