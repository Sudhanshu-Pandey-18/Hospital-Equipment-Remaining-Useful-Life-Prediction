# Hospital Equipment Remaining Useful Life Prediction

## Overview
This project focuses on predicting how long a hospital equipment can continue to function before it is likely to fail. The idea is to help hospitals plan maintenance in advance instead of reacting after a breakdown happens.

The model is trained using historical data such as equipment age, number of maintenance activities, downtime, and error frequency. A simple web interface is built using Streamlit so users can input values and get predictions instantly.

---

## Problem Statement
Unexpected failure of medical equipment can lead to serious operational issues in hospitals. Regular maintenance is important, but it is often not optimized.

This project aims to solve that by predicting the **remaining useful life (RUL)** of equipment so that maintenance decisions can be more data-driven.

---

## Features
- Predicts remaining life of equipment in years  
- Simple and interactive UI using Streamlit  
- Risk classification (Low, Moderate, High)  
- Basic visualization of input conditions  
- Model explainability using SHAP  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- SHAP  

---

---

## How It Works
- User enters equipment details from the sidebar  
- Data is processed to match the trained model format  
- Model predicts remaining useful life  
- Based on prediction and input values, a risk level is shown  
- SHAP is used to explain how each feature affected the prediction  

---


---

## Notes
This project is built as a machine learning application to demonstrate predictive maintenance in healthcare equipment. It can be further extended with real-time data or deployment on cloud platforms.

---

## Author
Sudhanshu Pandey
