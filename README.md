# Corneal AI Project — Starter Implementation

This is a research/learning prototype for the project:
“AI-Based Prediction of Grading of Hypothermia-Induced Donor Corneal Endothelial Damage and Recovery”.

## Important limitations
- This starter does NOT diagnose corneal health or decide transplant suitability.
- The classifier is not medically valid until trained and evaluated using a suitable, expert-labelled microscopy dataset.
- Do not label images as Mild/Moderate/Severe by guessing. Use labels supplied by your project guide/lab.
- The demo temperature mode is simulated. It is not real storage monitoring.
- A damage-grade classifier alone does not predict biological recovery. Recovery prediction requires separately labelled recovery data.

## 1. Open the project
Extract the ZIP. In VS Code choose File > Open Folder and select `corneal_ai_project`.

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
Then install packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add labelled images
Put your microscopy images in `dataset/images/`.
Copy `dataset/labels_template.csv` to `dataset/labels.csv`.
Fill it with one row per image, using the exact filenames and labels assigned by your supervisor/lab. Example columns:
`filename,grade`
`sample001.png,Mild`
`sample002.png,Moderate`
`sample003.png,Severe`

The example labels above explain the format only; they are NOT clinical thresholds.

## 4. Train the model
```powershell
python train_model.py
```
The script extracts basic image features and trains a Random Forest classifier. It saves `models/damage_model.joblib` and reports a validation score when the dataset is large enough. Very small datasets cannot provide reliable performance estimates.

## 5. Run the dashboard
```powershell
streamlit run app.py
```
Open the local address Streamlit prints in the terminal.

Until a trained model exists, the dashboard can show image preprocessing and features, but it will not invent a damage grade.

## 6. Temperature mode
The dashboard defaults to simulated temperature for UI testing. Select Serial (Arduino) after connecting the board and choosing the correct COM port. Arduino code is in `arduino/temp_monitor/temp_monitor.ino`.

Install the TMP117 Arduino library from Arduino IDE Library Manager (search `Adafruit TMP117`) and `Adafruit BusIO` if prompted. The sketch sends one temperature value per line over USB serial at 9600 baud. The dashboard reads the most recent value.

Hardware wiring depends on your exact TMP117 breakout and LCD module. Check their pin labels and voltage requirements before connecting. The included sketch uses I2C for TMP117 and a common I2C LCD address `0x27`; adjust it for your module. Do not use this prototype to monitor actual donor tissue.

## 7. Folder structure
- `app.py` — Streamlit dashboard
- `image_processing.py` — preprocessing and feature extraction
- `train_model.py` — model training
- `dataset/images/` — add images here
- `dataset/labels_template.csv` — label file template
- `models/` — trained model saved here
- `arduino/temp_monitor/temp_monitor.ino` — Arduino temperature code
