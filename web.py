from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)


with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

data = pd.read_csv('cleaned_cardekho.csv')
data_features = data.reindex(columns=feature_columns, fill_value=0)
data_features = data_features.fillna(0)
data_pred_log = loaded_model.predict(data_features)
data_pred_price = np.expm1(data_pred_log)


original_data = pd.read_csv('cardekho.csv')

def format_user_input(user_input_dict, feature_columns):
    user_input_df = pd.DataFrame([user_input_dict])
    for col in feature_columns:
        if col not in user_input_df:
            user_input_df[col] = 0
    user_input_df = user_input_df[feature_columns]
    return user_input_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_get', methods=['GET'])
def predict_get():
    year_str = request.args.get('year')
    km_driven_str = request.args.get('km_driven')
    fuel_type = request.args.get('fuel_type', '').strip().capitalize()
    transmission = request.args.get('transmission', '').strip().capitalize()
    owner = request.args.get('owner', '').strip().capitalize()
    mileage_str = request.args.get('mileage')
    engine_str = request.args.get('engine')
    max_power_str = request.args.get('max_power')
    seats_str = request.args.get('seats')
    brand = request.args.get('brand', '').strip().capitalize()

    if (year_str is None or km_driven_str is None or mileage_str is None or 
        engine_str is None or max_power_str is None or seats_str is None):
        return render_template('results.html', error="Missing parameters. Please fill in all fields.")

    try:
        year = int(year_str)
        km_driven = float(km_driven_str)
        mileage = float(mileage_str)
        engine = float(engine_str)
        max_power = float(max_power_str)
        seats = int(seats_str)
    except ValueError:
        return render_template('results.html', error="Invalid parameter types. Please enter values in the correct format.")

    current_year = 2024
    car_age = current_year - year

    input_dict = {
        'km_driven': km_driven,
        'mileage(km/ltr/kg)': mileage,
        'engine': engine,
        'car_age': car_age,
        'max_power': max_power,
        'seats': seats,
    }

    # One-hot encoding columns
    for col in feature_columns:
        if col.startswith('fuel_'):
            input_dict[col] = 1 if f'fuel_{fuel_type}' == col else 0
        elif col.startswith('transmission_'):
            input_dict[col] = 1 if f'transmission_{transmission}' == col else 0
        elif col.startswith('owner_'):
            input_dict[col] = 1 if f'owner_{owner}' == col else 0
        elif col.startswith('brand_'):
            normalized_brand = brand.replace('-', ' ').title()
            input_dict[col] = 1 if f'brand_{normalized_brand}' == col else 0

    user_input_df = format_user_input(input_dict, feature_columns)

    # Price prediction
    predicted_price_log = loaded_model.predict(user_input_df)
    predicted_price = np.expm1(predicted_price_log)[0]

    # Similar car recommendations
    distances = euclidean_distances(user_input_df, data_features)
    distances = distances.flatten()
    recommended_idx = np.argsort(distances)[:5]

    recommendations = []
    for i in recommended_idx:
        recommended_car_info = original_data.iloc[i].to_dict()
        actual_pred_price = data_pred_price[i]
        actual_selling_price = recommended_car_info['selling_price']

        recommended_car_info['predicted_price'] = round(float(actual_pred_price), 2)
        recommended_car_info['actual_selling_price'] = round(float(actual_selling_price), 2)
        recommendations.append(recommended_car_info)

    return render_template('results.html',
        predicted_price=round(float(predicted_price), 2),
        recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
