

# BTC-Price-Prediction-ML-Project

## About
This project focuses on the prediction of the prices of Bitcoin, the most in-demand cryptocurrency of today's world. We predict the prices accurately by gathering data available at **[Binance](https://data.binance.com/)** while taking various hyper-parameters into consideration which have affected the Bitcoin prices until now. This comprehensive project implements **13 different machine learning models** ranging from traditional statistical approaches to advanced deep learning techniques.

## Paper presentation
The [paper](9.pdf) contains all details of algorithms used along with results, analysis and discussions on the topic.

### Dataset
* Dataset has been downloaded using the **Binance API** with 365-day historical data.
* Real-time data fetching with OHLCV (Open, High, Low, Close, Volume) features.

* Dataset after Preprocessing

![Dataset after preprocessing](imgs/df. the Time-Series after order-1 differencing (to make it stationary)

![Seasonal Decomposition After Order-1 Differencing](imgs/ 



### Model Results & Performance

* **Champion Model - SVR (Support Vector Regression)** 

![SVR Results](imgs/svARCH-SARIMAX Model (Advanced Volatility Modeling)**

![GARCH-SARIMAX Results](imgs/garch- Regression (Non-linear Excellence)**

![Polynomial Results](imgs/polynomialBayesian Regression (Probabilistic Approach)**



* **LSTM Deep Learning Model**

![LSTM Results](imgs/lstm-results.png

![Random Forest Results]( Models Used:

#### Statistical Time Series Models:
* AR (AutoRegressive)
* ARMA (AutoRegressive Moving Average)  
* ARIMA (AutoRegressive Integrated Moving Average)
* SARIMAX (Seasonal ARIMA with eXogenous variables)
* GARCH-SARIMAX (Volatility modeling with SARIMAX residuals)
* VAR (Vector Autoregression)
* Auto-ARIMA (Automated parameter selection)
* Bayesian Regression with optimal parameters

#### Machine Learning Models:
* Polynomial Regression with feature engineering
* Elastic Net (L1 + L2 regularization)
* Support Vector Regression (SVR) with RBF kernel
* Random Forest ensemble method

#### Deep Learning Models:
* LSTM (Long Short-Term Memory) neural networks

#### Specialized Models:
* Prophet (Facebook's time series forecasting)

### Python Dependencies:
* pandas
* numpy
* requests
* matplotlib
* statsmodels
* pmdarima
* arch
* scikit-learn
* tensorflow
* prophet
* python-binance

### Install Dependencies (requirements.txt)
1. pip install -r requirements.txt

OR

1. pipenv install --ignore
2. pipenv shell

### How to Run
1. cd \<PROJECT ROOT DIRECTORY\>
2. python \<filename\>.py

### File Descriptions:
* auto-ARIMA.py: Runs automated gridsearch from pmdarima library, to find the best model parameters.
* AR.py, ARMA.py, ARIMA.py, SARIMAX.py: Use the above found best parameters to train the respective models as per their filenames.
* GARCH-SARIMAX.py: Runs SARIMAX models added with error of residuals from SARIMAX using GARCH.
* elasticnet.py: Runs Linear Regression with a combination of L1 and L2 penalty.
* bayesian.py: Runs BayesianRidge regression with optimal parameters.
* polyreg.py: Runs Linear Regression by adding polynomial features.
* svr_model.py: Support Vector Regression with RBF kernel - **Champion Model**.
* random_forest.py: Ensemble method using multiple decision trees.
* lstm.py: Deep learning implementation using LSTM neural networks.
* prophet_model.py: Facebook's specialized time series forecasting model.
* var.py: Vector Autoregression for multivariate time series analysis.

### Model Performance Rankings:
| Rank | Model | Performance | Specialization |
|------|-------|-------------|----------------|
| 🥇 1st | **SVR** | Excellent | Perfect correlation tracking |
| 🥈 2nd | **GARCH-SARIMAX** | Excellent | Volatility modeling expert |
| 🥉 3rd | **Polynomial** | Excellent | Non-linear pattern recognition |
| 4th | **Bayesian** | Very Good | Uncertainty quantification |
| 5th | **ARIMA** | Good | Statistical foundation (RMSE: 178.20) |
| 6th | **Random Forest** | Good | Ensemble learning |
| 7th | **SARIMAX** | Good | Seasonal pattern detection |

### Project Achievements:
- ✅ **13 Different Model Types** successfully implemented
- ✅ **Live Binance API Integration** with real-time data  
- ✅ **365-Day Standardized Dataset** for fair comparison
- ✅ **Professional Visualizations** for each model
- ✅ **Performance Benchmarking** with RMSE metrics
- ✅ **Industry-Level Implementation** ready for production

**Success Rate: 93% (13/14 models working)**

