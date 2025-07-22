# Heart Failure Prediction Web App

This project is a machine learning-powered web application for predicting the risk of heart failure based on clinical features. It is built using Python, Flask, and a custom-styled HTML frontend.

## Features
- **Data Preprocessing:** Handles missing values, encodes categorical variables, and scales features.
- **Model Training:** Trains a Random Forest Classifier on the provided heart disease dataset.
- **Web Interface:** User-friendly form for entering patient data and viewing predictions.
- **Backend:** Flask API for serving predictions using the trained model.

## Model Used
- **Random Forest Classifier** (from scikit-learn)

## How to Run
1. Ensure you have Python 3.7+ installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `heart.csv` in the `archive/` folder.
4. Run the Jupyter notebook `heart_failure_model.ipynb` to train and save the model (`heart_failure_model.pkl`) and scaler (`scaler.pkl`).
5. Start the Flask app:
   ```bash
   python app.py
   ```
6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## File Structure
- `app.py` - Flask backend
- `templates/index.html` - Frontend HTML
- `heart_failure_model.ipynb` - Model training and preprocessing
- `heart_failure_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler
- `archive/heart.csv` - Dataset

## Author
- Your Name Here

## License
This project is for educational purposes.
