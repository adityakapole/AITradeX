import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from utils.llm_handler import sentiment_analyzer
from config.settings import *
# this is a predictive analysis module that uses time series forecasting models to predict stock prices.
warnings.filterwarnings('ignore')

class PredictiveAnalyzer:
    def __init__(self):
        self.data = None
        self.arima_model = None
        self.sarima_model = None
        self.prophet_model = None
        self.arima_forecast = None
        self.sarima_forecast = None
        self.prophet_forecast = None
        self.model_metrics = {}
        
    def load_data(self, stock_data):
        """Load and prepare the stock data"""
        try:
            # Use the stock data directly
            self.data = stock_data['Close'].resample('D').mean().dropna()
            return True
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    def train_arima(self, order=(1,1,1)):
        """Train ARIMA model"""
        try:
            self.arima_model = ARIMA(self.data, order=order)
            self.arima_model = self.arima_model.fit()
            return True
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            return False
    
    def train_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Train SARIMA model"""
        try:
            self.sarima_model = SARIMAX(self.data, order=order, seasonal_order=seasonal_order)
            self.sarima_model = self.sarima_model.fit()
            return True
        except Exception as e:
            print(f"Error training SARIMA model: {e}")
            return False
    
    def train_prophet(self):
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': self.data.index,
                'y': self.data.values
            }).reset_index(drop=True)  # Reset index to ensure proper dimensionality
            
            self.prophet_model = Prophet(daily_seasonality=True)
            self.prophet_model.fit(prophet_data)
            return True
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            return False
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        # Split data into train and test
        train_size = int(len(self.data) * 0.8)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        # ARIMA evaluation
        if self.arima_model:
            arima_pred = self.arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
            self.model_metrics['ARIMA'] = {
                'MSE': mean_squared_error(test_data, arima_pred),
                'MAE': mean_absolute_error(test_data, arima_pred),
                'R2': r2_score(test_data, arima_pred)
            }
        
        # SARIMA evaluation
        if self.sarima_model:
            sarima_pred = self.sarima_model.predict(start=test_data.index[0], end=test_data.index[-1])
            self.model_metrics['SARIMA'] = {
                'MSE': mean_squared_error(test_data, sarima_pred),
                'MAE': mean_absolute_error(test_data, sarima_pred),
                'R2': r2_score(test_data, sarima_pred)
            }
        
        # Prophet evaluation
        if self.prophet_model:
            future = self.prophet_model.make_future_dataframe(periods=len(test_data))
            prophet_forecast = self.prophet_model.predict(future)
            prophet_pred = prophet_forecast['yhat'].values[-len(test_data):]
            self.model_metrics['Prophet'] = {
                'MSE': mean_squared_error(test_data, prophet_pred),
                'MAE': mean_absolute_error(test_data, prophet_pred),
                'R2': r2_score(test_data, prophet_pred)
            }
    
    def generate_forecasts(self, forecast_days):
        """Generate forecasts for all models"""
        # ARIMA forecast
        if self.arima_model:
            self.arima_forecast = self.arima_model.forecast(steps=forecast_days)
        
        # SARIMA forecast
        if self.sarima_model:
            self.sarima_forecast = self.sarima_model.forecast(steps=forecast_days)
        
        # Prophet forecast
        if self.prophet_model:
            future = self.prophet_model.make_future_dataframe(periods=forecast_days)
            prophet_forecast = self.prophet_model.predict(future)
            self.prophet_forecast = prophet_forecast['yhat'].values[-forecast_days:]
    
    def analyze_model_fit(self):
        """Analyze if models are underfitting or overfitting"""
        analysis = {}
        
        for model_name, metrics in self.model_metrics.items():
            r2 = metrics['R2']
            mse = metrics['MSE']
            
            if r2 > 0.95:
                fit_status = "Perfect Fit"
            elif r2 > 0.8:
                fit_status = "Good Fit"
            elif r2 > 0.6:
                fit_status = "Moderate Fit"
            else:
                fit_status = "Poor Fit"
            
            analysis[model_name] = {
                'R2': r2,
                'MSE': mse,
                'Fit Status': fit_status
            }
        
        return analysis
    
    def plot_forecasts(self):
        """Plot actual data and forecasts"""
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data.values, label='Actual', color='black')
        
        if self.arima_forecast is not None:
            plt.plot(self.arima_forecast.index, self.arima_forecast.values, 
                    label='ARIMA Forecast', color='blue', linestyle='--')
        
        if self.sarima_forecast is not None:
            plt.plot(self.sarima_forecast.index, self.sarima_forecast.values, 
                    label='SARIMA Forecast', color='red', linestyle='--')
        
        if self.prophet_forecast is not None:
            future_dates = pd.date_range(start=self.data.index[-1], 
                                       periods=len(self.prophet_forecast)+1)[1:]
            plt.plot(future_dates, self.prophet_forecast, 
                    label='Prophet Forecast', color='green', linestyle='--')
        
        plt.title('Stock Price Forecasts')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('forecast_plot.png')
        plt.close()
    
    def generate_analysis_report(self, invest_duration):
        """Generate comprehensive analysis report"""
        report = []
        report.append("=== Predictive Analysis Report ===\n")
        
        # Model Performance
        report.append("Model Performance Metrics:")
        for model_name, metrics in self.model_metrics.items():
            report.append(f"\n{model_name}:")
            report.append(f"R² Score: {metrics['R2']:.4f}")
            report.append(f"Mean Squared Error: {metrics['MSE']:.4f}")
            report.append(f"Mean Absolute Error: {metrics['MAE']:.4f}")
        
        # Model Fit Analysis
        fit_analysis = self.analyze_model_fit()
        report.append("\nModel Fit Analysis:")
        for model_name, analysis in fit_analysis.items():
            report.append(f"\n{model_name}:")
            report.append(f"R² Score: {analysis['R2']:.4f}")
            report.append(f"Fit Status: {analysis['Fit Status']}")
        
        # Price Predictions
        report.append("\nPrice Predictions:")
        if self.arima_forecast is not None:
            report.append(f"\nARIMA Model Prediction ({invest_duration} days):")
            report.append(f"Predicted Price Range: ${self.arima_forecast.min():.2f} - ${self.arima_forecast.max():.2f}")
            report.append(f"Final Predicted Price: ${self.arima_forecast.iloc[-1]:.2f}")
        
        if self.sarima_forecast is not None:
            report.append(f"\nSARIMA Model Prediction ({invest_duration} days):")
            report.append(f"Predicted Price Range: ${self.sarima_forecast.min():.2f} - ${self.sarima_forecast.max():.2f}")
            report.append(f"Final Predicted Price: ${self.sarima_forecast.iloc[-1]:.2f}")
        
        if self.prophet_forecast is not None:
            report.append(f"\nProphet Model Prediction ({invest_duration} days):")
            report.append(f"Predicted Price Range: ${self.prophet_forecast.min():.2f} - ${self.prophet_forecast.max():.2f}")
            report.append(f"Final Predicted Price: ${self.prophet_forecast[-1]:.2f}")
        
        # Model Selection Recommendation
        best_model = max(self.model_metrics.items(), key=lambda x: x[1]['R2'])[0]
        report.append(f"\nBest Performing Model: {best_model}")
        
        return "\n".join(report)

    def generate_final_llm_analysis(self, ticker, invest_duration):
        """Generate final LLM analysis of predictive results"""
        analysis_text = f"""
        Based on the following predictive analysis results for {ticker} over {invest_duration} days:

        Model Performance:
        {self.generate_analysis_report(invest_duration)}

        Please provide a final, decisive recommendation combining all predictive models.
        Include specific price targets, confidence levels, and risk management parameters.
        """

        final_analysis = sentiment_analyzer.analyze_sentiment(
            stock_ticker=ticker,
            text=analysis_text,
            invest_duration=invest_duration
        )
        
        return final_analysis

def analyze_predictions(ticker: str, invest_duration: int, stock_data: pd.DataFrame) -> str:
    """
    Main function to perform predictive analysis
    
    Args:
        ticker (str): Stock symbol
        invest_duration (int): Investment duration in days
        stock_data (pd.DataFrame): Historical stock data
        
    Returns:
        str: Analysis report
    """
    analyzer = PredictiveAnalyzer()
    
    # Load data
    if not analyzer.load_data(stock_data):
        return "Error: Failed to prepare stock data"
    
    # Train models
    analyzer.train_arima()
    analyzer.train_sarima()
    analyzer.train_prophet()
    
    # Evaluate models
    analyzer.evaluate_models()
    
    # Generate forecasts
    analyzer.generate_forecasts(invest_duration)
    
    # Plot forecasts
    analyzer.plot_forecasts()
    
    # Generate analysis report
    report = analyzer.generate_analysis_report(invest_duration)
    
    # Generate final LLM analysis
    final_analysis = analyzer.generate_final_llm_analysis(ticker, invest_duration)
    
    # Combine reports
    combined_report = f"{report}\n\n=== Final Expert Analysis ===\n{final_analysis}"
    
    return combined_report

if __name__ == "__main__":
    # Example usage
    ticker = "TSLA"
    invest_duration = 30
    stock_data = pd.read_csv('stock_data.csv', skiprows=1)
    stock_data.index = pd.to_datetime(stock_data.iloc[:, 0])
    stock_data = stock_data['Close'].resample('D').mean().dropna()
    report = analyze_predictions(ticker, invest_duration, stock_data)
    print(report) 