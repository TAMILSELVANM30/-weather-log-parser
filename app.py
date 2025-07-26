from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import json
from scipy import stats
from datetime import datetime

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Enhanced Comfort Classification with Heat Index
def calculate_heat_index(temp, humidity):
    """Calculate simplified heat index"""
    return temp + 0.5 * (humidity / 100) * (temp - 14.3)

def classify_comfort(temp, humidity):
    try:
        temp = float(temp)
        humidity = float(humidity)
    except:
        return "Data Error ❌"

    # Heat Index Consideration
    heat_index = calculate_heat_index(temp, humidity)
    if heat_index > 35:
        return "Dangerous Heat Index ⚠️"

    # Wind Chill Factor
    if temp < 10:
        wind_chill = temp - (0.7 * humidity/100)
        if wind_chill < -5:
            return "Bitter Wind Chill 🌬️"

    if temp > 45 or humidity > 90:
        return "Extremely Uncomfortable 🔥"
    elif temp > 35 and humidity > 70:
        return "Very Uncomfortable 🥵"
    elif temp < 5:
        return "Freezing Cold 🧊"
    elif temp < 15 and humidity < 30:
        return "Chilly and Dry ❄️"
    elif 20 <= temp <= 30 and 40 <= humidity <= 60:
        return "Comfortable 😊"
    elif 15 <= temp <= 20 and 30 <= humidity <= 50:
        return "Mild and Pleasant 🌤️"
    else:
        return "Moderately Uncomfortable 😓"

def deep_analysis(df):
    # Validation Checks
    if df.empty:
        # Ensure all return values are consistent
        return None, None, None, None, None, None, "❌ CSV file is empty"

    df.columns = [col.lower().strip() for col in df.columns]
    required_cols = ['temperature', 'humidity']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Ensure all return values are consistent
        return None, None, None, None, None, None, f"❌ Missing columns: {', '.join(missing_cols)}"

    try:
        df['temperature'] = pd.to_numeric(df['temperature'])
        df['humidity'] = pd.to_numeric(df['humidity'])
    except ValueError:
        # Ensure all return values are consistent
        return None, None, None, None, None, None, "❌ Invalid data in Temperature/Humidity columns"

    df['comfort_index'] = df.apply(lambda row: classify_comfort(row['temperature'], row['humidity']), axis=1)

    # Statistical Analysis
    df['temp_zscore'] = np.abs(stats.zscore(df['temperature']))
    df['humidity_zscore'] = np.abs(stats.zscore(df['humidity']))

    summary = {
        "max_temp": round(df['temperature'].max(), 2),
        "min_temp": round(df['temperature'].min(), 2),
        "avg_temp": round(df['temperature'].mean(), 2),
        "max_humidity": round(df['humidity'].max(), 2),
        "min_humidity": round(df['humidity'].min(), 2),
        "avg_humidity": round(df['humidity'].mean(), 2),
        "hot_day_alert": df['temperature'].max() >= 40,
        "cold_day_alert": df['temperature'].min() <= 5,
        "most_common_comfort": df['comfort_index'].value_counts().idxmax(),
        "heat_wave": 3 if len(df[df['temperature'] > 35]) >= 3 else 0,
        "comfort_distribution": df['comfort_index'].value_counts().to_dict(),
        "temp_std": round(df['temperature'].std(), 2),
        "humidity_std": round(df['humidity'].std(), 2)
    }

    conditions = df['condition'].value_counts().to_dict() if 'condition' in df.columns else None

    # Anomaly Detection
    anomalies = []
    extreme_days = []
    z_anomalies = []

    # Initialize chart_data here, so it always has a value
    chart_data = {
        "labels": [],
        "temperature": [],
        "humidity": []
    }

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)

        # Temperature Change Anomalies
        df['temp_diff'] = df['temperature'].diff().fillna(0)
        df['abs_diff'] = df['temp_diff'].abs()
        anomaly_df = df[(df['abs_diff'] > 5) & (df['abs_diff'] <= 15)]
        anomalies = anomaly_df[['date', 'temp_diff']].rename(columns={'temp_diff': 'temperature_change'}).to_dict(orient='records')

        # Z-Score Anomalies
        z_anomalies = df[(df['temp_zscore'] > 2) | (df['humidity_zscore'] > 2)]
        z_anomalies = z_anomalies[['date', 'temperature', 'humidity']].rename(columns={
            'temperature': 'temp_value',
            'humidity': 'humidity_value'
        }).to_dict(orient='records')

        # Extreme Days Detection
        for _, row in df.iterrows():
            label = ""
            if row['temperature'] >= 40:
                label = "🔥 Extremely Hot"
            elif row['temperature'] <= 5:
                label = "🧊 Freezing"
            elif row['humidity'] >= 90:
                label = "💦 Very Humid"
            elif row['comfort_index'] == "Extremely Uncomfortable 🔥":
                label = "🥵 Overheat"
            if label:
                extreme_days.append({
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "temp": row['temperature'],
                    "humidity": row['humidity'],
                    "label": label
                })

        # Chart Data Preparation (reassigned if 'date' column exists)
        chart_data = {
            "labels": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "temperature": df['temperature'].tolist(),
            "humidity": df['humidity'].tolist()
        }

    # IMPORTANT: Do not json.dumps here. Pass the Python dictionary.
    # json.dumps should happen in the Flask route right before rendering.
    return summary, conditions, anomalies, extreme_days, z_anomalies, chart_data, None

@app.route("/", methods=["GET", "POST"])
def index():
    summary = conditions = anomalies = extreme_days = z_anomalies = chart_data = message = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                file.save(filepath)
                df = pd.read_csv(filepath)
                # deep_analysis now returns chart_data as a Python dict
                summary, conditions, anomalies, extreme_days, z_anomalies, chart_data_raw, error = deep_analysis(df)
                if error:
                    message = error
                # Only dump chart_data to JSON just before passing to template
                chart_data = json.dumps(chart_data_raw, default=str)
            except Exception as e:
                message = f"❌ Error processing file: {str(e)}"

    return render_template(
        "index.html",
        summary=summary,
        conditions=conditions,
        anomalies=anomalies,
        extreme_days=extreme_days,
        z_anomalies=z_anomalies,
        chart_data=chart_data, # Pass the JSON string to the template
        message=message
    )

@app.route('/export')
def export_data():
    # Implement your export logic here
    return send_file('path/to/processed_data.csv',
                     mimetype='text/csv',
                     download_name='weather_analysis.csv',
                     as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)