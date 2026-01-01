"""
Disease Prediction Flask Application
=====================================
A web application for predicting diseases using a Random Forest model.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import io
import base64

from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_FILENAME = 'random_forest_model.pkl'
ALLOWED_EXTENSIONS = {'csv'}

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, MODEL_FILENAME)
model = joblib.load(model_path)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_confusion_matrix(y_true: pd.Series, y_pred) -> str:
    """
    Generate a confusion matrix heatmap and return it as a base64 string.
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
    
    Returns:
        Base64 encoded PNG image string
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Create figure with a dark background
    plt.figure(figsize=(12, 9))
    plt.style.use('default')
    
    # Create heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64


@app.route('/')
def home():
    """Render the home page with file upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle file upload and return predictions.
    
    Expects a CSV file with:
    - Feature columns matching the training data
    - A 'prognosis' column with actual labels for accuracy calculation
    
    Returns:
        Rendered result template with accuracy and confusion matrix
    """
    # Validate file upload
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded"), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="No file selected"), 400
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Please upload a CSV file."), 400
    
    try:
        # Read and process the CSV file
        data = pd.read_csv(file)
        
        # Validate required columns
        if 'prognosis' not in data.columns:
            return render_template(
                'index.html', 
                error="CSV file must contain a 'prognosis' column"
            ), 400
        
        # Prepare features and labels
        X_test = data.drop(['prognosis'], axis=1)
        y_test = data['prognosis']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = (y_test == y_pred).mean() * 100
        
        # Generate confusion matrix image
        img_base64 = generate_confusion_matrix(y_test, y_pred)
        
        return render_template(
            'result.html',
            accuracy=f"{accuracy:.2f}",
            img_base64=img_base64
        )
        
    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error processing file: {str(e)}"
        ), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('index.html', error="Internal server error"), 500


if __name__ == "__main__":
    print("=" * 50)
    print("Disease Prediction Web Application")
    print("=" * 50)
    print(f"Model loaded: {MODEL_FILENAME}")
    print("Starting server on http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
