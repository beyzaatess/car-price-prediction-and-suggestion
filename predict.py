import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def format_user_input(user_input_dict, feature_columns):
    user_input_df = pd.DataFrame([user_input_dict])
    for col in feature_columns:
        if col not in user_input_df:
            user_input_df[col] = 0
    user_input_df = user_input_df[feature_columns]
    return user_input_df

def load_model_and_predict():

    with open('trained_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    print("\n--- Collect User Input ---")
    year = int(input("Vehicle manufacturing year: "))
    km_driven = float(input("Kilometers driven by the vehicle (e.g., 50000): "))
    fuel_type = input("Fuel type (Diesel, Petrol, CNG, LPG): ").strip().capitalize()
    transmission = input("Transmission type (Manual, Automatic): ").strip().capitalize()
    owner = input("Ownership type (First Owner, Second Owner, Third Owner, etc.): ").strip().capitalize()
    mileage = float(input("Mileage (e.g., 21.4): "))
    engine = float(input("Engine capacity (e.g., 1248): "))
    max_power = float(input("Maximum power (e.g., 74): "))
    seats = int(input("Number of seats in the vehicle (e.g., 5): "))
    brand = input("Vehicle brand (e.g., Maruti, Hyundai): ").strip().capitalize()

    current_year = 2024
    car_age = current_year - year

    user_input = {
        'km_driven': km_driven,
        'mileage(km/ltr/kg)': mileage,
        'engine': engine,
        'car_age': car_age,
        'max_power': max_power,
        'seats': seats,
    }

    for col in feature_columns:
        if col.startswith('fuel_'):
            user_input[col] = 1 if f'fuel_{fuel_type}' == col else 0
        elif col.startswith('transmission_'):
            user_input[col] = 1 if f'transmission_{transmission}' == col else 0
        elif col.startswith('owner_'):
            user_input[col] = 1 if f'owner_{owner}' == col else 0
        elif col.startswith('brand_'):
            normalized_brand = brand.replace('-', ' ').title()
            user_input[col] = 1 if f'brand_{normalized_brand}' == col else 0

    user_input_df = format_user_input(user_input, feature_columns)

    predicted_price_log = loaded_model.predict(user_input_df)
    predicted_price = np.expm1(predicted_price_log)
    print(f"\nPredicted car price: {predicted_price[0]:,.2f} units.\n")

    
    data_cleaned = pd.read_csv('cleaned_cardekho.csv')
    data_original = pd.read_csv('cardekho.csv') 

    data_features = data_cleaned.reindex(columns=feature_columns, fill_value=0)

    if data_features.isnull().sum().sum() > 0:
        data_features = data_features.fillna(0)

    data_pred_log = loaded_model.predict(data_features)
    data_pred_price = np.expm1(data_pred_log)

    distances = euclidean_distances(user_input_df, data_features)
    distances = distances.flatten()

    recommended_idx = np.argsort(distances)[:5]

    print("Top 5 Most Similar Cars:\n")
    
    recommendations = []

    for i in recommended_idx:
        print("--------------------------------------------------------")
        print("RECOMMENDED CAR INFORMATION:\n")
        
        recommended_car_info_original = data_original.iloc[i]
        actual_pred_price = data_pred_price[i]
        actual_selling_price = np.expm1(data_cleaned.iloc[i]['selling_price'])
        
        print(recommended_car_info_original.to_string())
        print(f"\nPredicted Price (model output): {actual_pred_price:,.2f} units")
        print(f"Actual Price (from dataset): {actual_selling_price:,.2f} units\n")

        recommendations.append({
            'original_index': i,
            'car_info': recommended_car_info_original.to_dict(),
            'predicted_price': float(actual_pred_price),
            'actual_price_in_dataset': float(actual_selling_price)
        })

    return recommendations

if __name__ == "__main__":
    recommended_cars = load_model_and_predict()

