# 🏦 Customer Churn Prediction Using Artificial Neural Networks

A machine learning project that predicts customer churn using Artificial Neural Networks (ANNs) with TensorFlow/Keras, featuring an interactive Streamlit web application for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Customer churn is a critical metric for businesses, especially in the banking and financial services sector. This project leverages deep learning to predict whether a customer will leave the bank based on various features such as credit score, geography, gender, age, tenure, balance, and more.

The project includes:
- **End-to-end machine learning pipeline** from data preprocessing to model deployment
- **Interactive web application** built with Streamlit for real-time predictions
- **Pre-trained model** and encoders for immediate use
- **Comprehensive notebooks** for experimentation and inference

---

## ✨ Features

- ✅ **Deep Learning Model**: Multi-layer ANN built with TensorFlow/Keras
- ✅ **Data Preprocessing**: Label encoding, one-hot encoding, and feature scaling
- ✅ **Real-time Predictions**: Interactive Streamlit dashboard
- ✅ **Model Persistence**: Saved model and encoders for deployment
- ✅ **User-friendly Interface**: Simple input fields with immediate feedback
- ✅ **Probability Scores**: Get churn probability percentages
- ✅ **Visual Feedback**: Color-coded results (Green for retained, Red for churn risk)

---

## 📊 Dataset

The project uses the **Churn Modelling Dataset** containing customer information from a bank:

**Features:**
- `CreditScore`: Customer's credit score
- `Geography`: Customer's location (France, Spain, Germany)
- `Gender`: Male or Female
- `Age`: Customer's age
- `Tenure`: Number of years with the bank
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used
- `HasCrCard`: Credit card ownership (0/1)
- `IsActiveMember`: Active membership status (0/1)
- `EstimatedSalary`: Estimated annual salary

**Target:**
- `Exited`: Whether the customer churned (1) or not (0)

**Dataset Size:** 10,000 customers

---

## 📁 Project Structure

```
AnnChurnModelling/
│
├── experiments.ipynb          # Model training and experimentation
├── prediction.ipynb           # Prediction examples and testing
├── app.py                     # Streamlit web application
│
├── model.h5                   # Trained ANN model
├── Scaler.pkl                 # StandardScaler for feature scaling
├── label_encoder_gender.pkl   # Label encoder for gender
├── OHE_encoder_geo.pkl        # One-hot encoder for geography
│
└── README.md                  # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sukrit-89/ANN_CHURN_MODELLING.git
   cd ANN_CHURN_MODELLING
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install streamlit tensorflow scikit-learn pandas numpy
   ```

---

## 💻 Usage

### Running the Web Application

Launch the Streamlit app for interactive predictions:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Notebooks

**For Training and Experimentation:**
```bash
jupyter notebook experiments.ipynb
```

**For Testing Predictions:**
```bash
jupyter notebook prediction.ipynb
```

### Making Predictions Programmatically

```python
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# Load model and encoders
model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('OHE_encoder_geo.pkl', 'rb') as f:
    ohe_encoder_geo = pickle.load(f)
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

# Process and predict
# ... (encoding and scaling steps)
prediction = model.predict(input_scaled)
```

---

## 🧠 Model Architecture

The Artificial Neural Network consists of:

- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layers**: Multiple dense layers with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation (binary classification)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

**Preprocessing Pipeline:**
1. Drop irrelevant columns (RowNumber, CustomerId, Surname)
2. Label encode Gender (Female=0, Male=1)
3. One-hot encode Geography (France, Germany, Spain)
4. Standard scaling of all features

---

## 📈 Results

The model achieves strong performance in predicting customer churn:

- 🎯 **Accurate Predictions**: Identifies at-risk customers effectively
- 📊 **Probability Scores**: Provides churn likelihood percentages
- ⚡ **Fast Inference**: Real-time predictions in the web app
- 🔄 **Reproducible**: Saved model and encoders ensure consistency

---

## 🖥️ Web Application

The Streamlit app provides an intuitive interface for making predictions:

### Features:
- **Interactive Input Fields**:
  - Dropdowns for categorical variables (Geography, Gender)
  - Sliders for bounded values (Age, Tenure, Number of Products)
  - Number inputs for continuous variables (Balance, Credit Score, Salary)

- **Instant Predictions**:
  - Click "Predict Churn" to get results
  - Color-coded output:
    - ✅ **Green**: Customer likely to stay
    - ⚠️ **Red**: Customer at risk of churning
  - Displays probability percentage

### Example Output:
```
⚠️ The customer is likely to churn (Probability: 87.32%)
```
or
```
✅ The customer is not likely to churn (Probability: 12.45%)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **TensorFlow/Keras** | Deep learning framework for building the ANN |
| **Scikit-learn** | Data preprocessing (encoding, scaling) |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Streamlit** | Interactive web application |
| **Pickle** | Model and encoder serialization |
| **Jupyter Notebook** | Experimentation and analysis |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Model optimization and hyperparameter tuning
- Additional feature engineering
- Enhanced UI/UX for the Streamlit app
- Model interpretability (SHAP, LIME)
- Integration with cloud deployment platforms
- Additional evaluation metrics and visualizations

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Sukrit**

- GitHub: [@sukrit-89](https://github.com/sukrit-89)
- Repository: [ANN_CHURN_MODELLING](https://github.com/sukrit-89/ANN_CHURN_MODELLING)

---

## 🙏 Acknowledgments

- Dataset source: Churn Modelling Dataset
- TensorFlow and Keras communities for excellent documentation
- Streamlit team for the amazing framework

---

## 📞 Contact

For questions, suggestions, or collaborations, feel free to reach out via GitHub issues!

---

<p align="center">
  <i>⭐ If you find this project helpful, please consider giving it a star! ⭐</i>
</p>

---

**Made with ❤️ and Python**
