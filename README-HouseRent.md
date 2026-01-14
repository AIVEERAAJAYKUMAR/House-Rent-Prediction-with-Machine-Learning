# House Rent Prediction with Machine Learning

An end-to-end machine learning project that predicts house rental prices using LSTM (Long Short-Term Memory) neural networks. The model analyzes various housing features to provide accurate rent predictions across different Indian cities.

## 📋 Table of Contents
- [Overview](##overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a deep learning solution for predicting house rental prices using historical housing data from various Indian cities. The model utilizes LSTM neural networks to capture temporal patterns and relationships between housing features and rental prices, making it suitable for real estate market analysis and rental price estimation.

**Key Objectives:**
- Predict rental prices based on property characteristics
- Analyze rental trends across different cities and BHK configurations
- Provide insights into factors affecting rental prices
- Build a robust deep learning model for regression tasks

## ✨ Features

- **Deep Learning Model**: LSTM-based neural network for accurate predictions
- **Multi-Feature Analysis**: Considers BHK, size, location, furnishing status, and more
- **Data Visualization**: Interactive plots using Plotly for exploratory analysis
- **Missing Value Handling**: Robust data preprocessing pipeline
- **Statistical Analysis**: Comprehensive descriptive statistics
- **City-wise Comparison**: Rental price analysis across multiple cities
- **Model Evaluation**: Performance metrics and validation

## 📊 Dataset

### Dataset Overview
- **Source**: House_Rent_Dataset.csv
- **Total Records**: 4,746 properties
- **Features**: 12 columns
- **Target Variable**: Rent (in INR)

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| Posted On | Date when property was listed | Date |
| BHK | Number of bedrooms (1-6) | Numerical |
| Rent | Monthly rental price in INR | Numerical (Target) |
| Size | Property area in sq. ft. | Numerical |
| Floor | Floor location (e.g., "Ground out of 2") | Categorical |
| Area Type | Super Area, Carpet Area, Built Area | Categorical |
| Area Locality | Specific neighborhood/area | Categorical |
| City | City name (Kolkata, Mumbai, Bangalore, etc.) | Categorical |
| Furnishing Status | Furnished, Semi-Furnished, Unfurnished | Categorical |
| Tenant Preferred | Bachelors, Family, Bachelors/Family | Categorical |
| Bathroom | Number of bathrooms (1-10) | Numerical |
| Point of Contact | Contact Owner, Contact Agent | Categorical |

### Dataset Statistics

```
Mean Rent: ₹34,993
Median Rent: ₹16,000
Highest Rent: ₹35,00,000
Lowest Rent: ₹1,200

BHK Distribution:
- Mean: 2.08 BHK
- Range: 1-6 BHK

Size Statistics:
- Mean: 967 sq. ft.
- Range: 10-8,000 sq. ft.
```

## 🛠 Technologies Used

### Programming & Libraries
- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static visualizations
- **Plotly**: Interactive visualizations
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Data preprocessing and scaling

### Development Environment
- **Google Colab**: Cloud-based Jupyter notebook environment
- **Jupyter Notebook**: Interactive development

### Python Libraries
```python
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
plotly==5.14.1
tensorflow==2.12.0
keras==2.12.0
scikit-learn==1.2.2
```

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Google Colab account

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-rent-prediction.git
   cd house-rent-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Place `House_Rent_Dataset.csv` in the project root directory

### Google Colab Setup

1. Upload the notebook to Google Colab
2. Upload the dataset file
3. Run all cells sequentially

## 💻 Usage

### Running the Complete Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv("House_Rent_Dataset.csv")

# Data exploration
print(data.head())
print(data.isnull().sum())
print(data.describe())

# Exploratory analysis
print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")

# Visualization
figure = px.bar(data, x=data["City"], y=data["Rent"], 
                color=data["BHK"],
                title="Rent in Different Cities According to BHK")
figure.show()
```

### Training the Model

The LSTM model follows this workflow:
1. Data preprocessing and normalization
2. Feature engineering and encoding
3. Train-test split (90-10)
4. Model training with specified architecture
5. Evaluation and prediction

## 🏗 Model Architecture

### LSTM Neural Network Structure

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 7, 128)            66,560    
_________________________________________________________________
lstm_1 (LSTM)                (None, 64)                49,408    
_________________________________________________________________
dense (Dense)                (None, 25)                1,625     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 26        
=================================================================
Total params: 117,619 (459.45 KB)
Trainable params: 117,619 (459.45 KB)
Non-trainable params: 0 (0.00 B)
```

### Model Configuration
- **Layer 1**: LSTM with 128 units, return sequences
- **Layer 2**: LSTM with 64 units
- **Layer 3**: Dense layer with 25 units
- **Output Layer**: Single unit for rent prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 21
- **Batch Size**: 1

### Training Details
- Training samples: 4,271
- Time per epoch: ~38-44 seconds
- Total training time: ~13-14 minutes

## 📈 Results

### Model Performance

The model successfully learns patterns in the housing rental data, with loss values indicating convergence:

```
Training Loss Progression:
Epoch 1: 5,366,150,144
Epoch 10: 2,395,378,688
Epoch 21: 2,265,982,976
```

### Key Insights

1. **City-wise Trends**: Rental prices vary significantly across cities
2. **BHK Impact**: Higher BHK correlates with increased rent
3. **Furnishing Premium**: Furnished properties command higher rents
4. **Size Matters**: Larger properties have proportionally higher rents
5. **Location Factor**: Area locality significantly affects pricing

## 📊 Visualizations

The project includes several interactive visualizations:

1. **Bar Chart**: Rent distribution across cities by BHK configuration
2. **Statistical Plots**: Distribution of key features
3. **Correlation Analysis**: Relationship between features and rent
4. **Time Series**: Trends in rental prices over time

## 🚀 Future Enhancements

- [ ] Implement additional models (Random Forest, XGBoost) for comparison
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Include more features (distance to metro, amenities)
- [ ] Create web application for rent prediction
- [ ] Implement model deployment with Flask/FastAPI
- [ ] Add geospatial visualization on maps
- [ ] Develop mobile-responsive dashboard
- [ ] Include seasonality analysis
- [ ] Add anomaly detection for outlier prices
- [ ] Implement cross-validation for robust evaluation



## 📧 Contact

**Veera Ajay Kumar Angajala**
- Email: ajay01.jobs@gmail.com
- LinkedIn: [Veera Angajala](https://linkedin.com/in/yourprofile)


## 🙏 Acknowledgments

- Dataset providers for making housing rental data publicly available
- TensorFlow and Keras teams for excellent deep learning frameworks
- Plotly team for interactive visualization capabilities
- Google Colab for providing free computational resources

## 📚 References

- [LSTM Networks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Real Estate Price Prediction Research](https://scholar.google.com/)

---

**⭐ If you found this project helpful, please consider giving it a star!**

*Last Updated: December 2025*
