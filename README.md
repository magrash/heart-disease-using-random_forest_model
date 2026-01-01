# 🧬 Disease Prediction using Random Forest

An AI-powered disease prediction web application built with Flask and scikit-learn. Upload patient symptom data in CSV format and receive instant predictions using a trained Random Forest classifier.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)

## ✨ Features

- **Machine Learning Prediction**: Uses a trained Random Forest model for disease classification
- **Web Interface**: Clean, modern UI for uploading CSV files
- **Confusion Matrix Visualization**: Visual representation of model performance
- **Accuracy Metrics**: Real-time accuracy calculation on uploaded data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/magrash/heart-disease-using-random_forest_model.git
   cd heart-disease-using-random_forest_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── random_forest_model.pkl # Trained model file
├── requirements.txt        # Python dependencies
├── Training.csv            # Training dataset
├── Testing.csv             # Testing dataset
├── static/
│   └── css/
│       └── style.css       # Application styling
└── templates/
    ├── index.html          # Upload page
    └── result.html         # Results page
```

## 📊 Model Training

To retrain the model with your own data:

```bash
python train_model.py
```

This will:
- Load training and testing data
- Train a Random Forest classifier
- Display accuracy metrics
- Save the model to `random_forest_model.pkl`

## 🔧 Usage

1. Prepare your CSV file with the same feature columns as the training data
2. Include a `prognosis` column with actual labels (for accuracy calculation)
3. Upload the file through the web interface
4. View the prediction accuracy and confusion matrix

## 📈 Model Performance

The Random Forest model achieves high accuracy on the test dataset. The confusion matrix visualization helps understand the model's predictions across different disease categories.

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **ML**: scikit-learn (Random Forest Classifier)
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism Design)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">Made with ❤️ using Machine Learning</p>
