@echo off
cd ..

python -m venv venv
call .\venv\Scripts\activate.bat

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0" --force-reinstall

echo.
echo Setup complete.
echo You can now run the scripts in this folder.
echo.
echo If you want to use accelerate, run the following command to configure it:
echo accelerate config
echo.

pause