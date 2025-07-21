# app.py
from flask import Flask, render_template, send_from_directory
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import os

app = Flask(__name__)

def generate_forecast():
    """
    This function contains the core logic from your original script.
    It fetches data, trains the model, makes a forecast, and saves the plot.
    """
    print("--- Generating new forecast ---")
    
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    start_date = datetime(1947, 1, 1)
    end_date = datetime.now()
    cpi_data = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

    cpi_quarterly = cpi_data['CPIAUCSL'].resample('QS').mean()
    inflation_rate = cpi_quarterly.pct_change(periods=4) * 100
    inflation_rate = inflation_rate.dropna()

    model = ARIMA(inflation_rate, order=(1, 1, 1))
    model_fit = model.fit()

    forecast_steps = 4
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast_values = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int()

    plt.figure(figsize=(14, 7))
    plt.plot(inflation_rate, label='Historical Inflation Rate')
    plt.plot(forecast_values, label='Forecasted Inflation Rate', color='red', linestyle='--')
    plt.fill_between(confidence_intervals.index,
                     confidence_intervals.iloc[:, 0],
                     confidence_intervals.iloc[:, 1], color='pink', alpha=0.5, label='95% Confidence Interval')
    plt.title('US Inflation Rate: Historical vs. Forecast')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(static_dir, 'forecast_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return forecast_values, plot_path

@app.route('/')
def index():
    """
    This function is called when a user visits the homepage.
    It calls the forecast generation function and renders the HTML page.
    """
    forecast_data, plot_filename = generate_forecast()
    
    formatted_forecast = []
    for date, value in forecast_data.items():
        formatted_forecast.append({
            'quarter': f"{date.year} Q{date.quarter}",
            'value': f"{value:.2f}%"
        })

    return render_template('index.html', 
                           forecasts=formatted_forecast, 
                           plot_url='forecast_plot.png')

@app.route('/static/<filename>')
def send_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    # For local development:
    # app.run(debug=True)
    # For production (like on Render), Gunicorn will be used.
    pass
