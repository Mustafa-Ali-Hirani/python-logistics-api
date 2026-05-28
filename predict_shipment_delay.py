# predict_shipment_delay.py
import os
import pandas as pd
import joblib

# ==========================================
# LOAD EXPORTED ML MODEL & ENCODERS
# ==========================================
MODEL_PATH = "delay_predictor_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    raise FileNotFoundError("Model or Encoder files not found. Please run 'train_delay_predictor.py' first.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

def predict_single_shipment(distance: int, weight: float, carrier: str, weather: str, congestion: str):
    """Encodes user input, feeds it to the trained classifier, and returns delay probability."""
    print(f"\n[Inference] Calculating delay risk for incoming shipment...")
    print(f" -> Carrier: {carrier} | Weather: {weather} | Congestion: {congestion} | Distance: {distance} mi")
    
    # 1. Validate and Encode Categorical Inputs
    try:
        encoded_carrier = encoders["carrier"].transform([carrier])[0]
        encoded_weather = encoders["weather"].transform([weather])[0]
        encoded_congestion = encoders["congestion"].transform([congestion])[0]
    except ValueError as e:
        print(f"\n[Error] Invalid category inputted: {e}")
        print(f"Valid carriers: {list(encoders['carrier'].classes_)}")
        print(f"Valid weather: {list(encoders['weather'].classes_)}")
        print(f"Valid congestion: {list(encoders['congestion'].classes_)}")
        return

    # 2. Structure input data exactly like the training feature set
    input_data = pd.DataFrame([{
        "distance_miles": distance,
        "weight_kg": weight,
        "carrier": encoded_carrier,
        "weather": encoded_weather,
        "congestion": encoded_congestion
    }])
    
    # 3. Predict Probability
    # predict_proba returns [prob_of_0, prob_of_1]
    probabilities = model.predict_proba(input_data)[0]
    delay_probability = probabilities[1]
    
    # 4. Predict binary class outcome (0 or 1)
    prediction = model.predict(input_data)[0]
    risk_level = "HIGH RISK" if delay_probability >= 0.50 else "LOW RISK"
    
    print("\n==========================================")
    print("         SHIPMENT DELAY PREDICTION        ")
    print("==========================================")
    print(f"Delay Probability: {delay_probability * 100:.2f}%")
    print(f"Assigned Status  : {risk_level}")
    print(f"Classification   : {'DELAY EXPECTED (1)' if prediction == 1 else 'ON-TIME EXPECTED (0)'}")
    print("==========================================")

if __name__ == "__main__":
    # Test with a High-Risk Storm Scenario
    predict_single_shipment(
        distance=6200,
        weight=18500.0,
        carrier="Swift Ocean Carriers",
        weather="Storm",
        congestion="High"
    )
    
    # Test with a Low-Risk Clear Scenario
    predict_single_shipment(
        distance=800,
        weight=4200.0,
        carrier="Swift Ocean Carriers",
        weather="Clear",
        congestion="Low"
    )