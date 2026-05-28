# train_delay_predictor.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =====================================================================
# STEP 1: GENERATE SYNTHETIC LOGISTICS DATASET
# =====================================================================
def generate_historical_shipments(n_samples=1500) -> pd.DataFrame:
    """Generates realistic historical logistics data with engineered delay patterns."""
    print(f"[Data Prep] Generating {n_samples} historical shipment records...")
    np.random.seed(42)  # For reproducible results
    
    # Generate random features
    distance_miles = np.random.randint(100, 8000, size=n_samples)
    shipment_weight_kg = np.random.uniform(500, 30000, size=n_samples)
    
    carriers = ["Swift Ocean Carriers", "Global Maritime Trust", "Pacific Cargo Group"]
    carrier_list = np.random.choice(carriers, size=n_samples)
    
    weather_conditions = ["Clear", "Rain", "Snow", "Storm"]
    weather_list = np.random.choice(weather_conditions, size=n_samples, p=[0.5, 0.3, 0.1, 0.1])
    
    congestion_levels = ["Low", "Medium", "High"]
    congestion_list = np.random.choice(congestion_levels, size=n_samples, p=[0.4, 0.4, 0.2])
    
    # Mathematical calculation of delay probability based on features (Ground Truth Logic)
    # This ensures the model has clear statistical patterns to learn
    delay_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.05  # Base delay probability is 5%
        
        # 1. Congestion impact
        if congestion_list[i] == "High":
            prob += 0.35
        elif congestion_list[i] == "Medium":
            prob += 0.15
            
        # 2. Weather impact
        if weather_list[i] == "Storm":
            prob += 0.45
        elif weather_list[i] == "Rain":
            prob += 0.10
            
        # 3. Distance impact
        if distance_miles[i] > 5000:
            prob += 0.15
            
        # 4. Carrier efficiency variance
        if carrier_list[i] == "Swift Ocean Carriers":
            prob -= 0.05  # Faster/more reliable carrier
            
        # Clip probability between 0 and 1
        delay_prob[i] = np.clip(prob, 0.0, 1.0)
        
    # Determine the binary target label (1 = delayed, 0 = on-time) based on the calculated probability
    delayed = np.random.binomial(1, delay_prob)
    
    # Assemble into a Pandas DataFrame
    df = pd.DataFrame({
        "distance_miles": distance_miles,
        "weight_kg": shipment_weight_kg,
        "carrier": carrier_list,
        "weather": weather_list,
        "congestion": congestion_list,
        "delayed": delayed
    })
    
    # Save the dataset to a CSV file for auditing
    df.to_csv("historical_shipments.csv", index=False)
    print("✓ Saved raw dataset to 'historical_shipments.csv'")
    return df

# =====================================================================
# STEP 2: PREPROCESS DATA (Encode categorical values to numbers)
# =====================================================================
def preprocess_data(df: pd.DataFrame):
    """Encodes string variables into numerical integers for training."""
    print("[Preprocessing] Encoding categorical labels...")
    
    # Copy DataFrame to avoid modifying original data
    processed_df = df.copy()
    
    # We use separate LabelEncoders so we can reuse them later to decode user inputs
    label_encoders = {}
    categorical_columns = ["carrier", "weather", "congestion"]
    
    for col in categorical_columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
        label_encoders[col] = le
        
    return processed_df, label_encoders

# =====================================================================
# STEP 3: TRAIN & EVALUATE MODEL
# =====================================================================
def train_model():
    # 1. Generate and preprocess
    raw_df = generate_historical_shipments()
    processed_df, encoders = preprocess_data(raw_df)
    
    # 2. Separate features (X) and target (y)
    X = processed_df.drop(columns=["delayed"])
    y = processed_df["delayed"]
    
    # 3. Perform Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"[Split] Training set size: {X_train.shape[0]} samples | Testing set size: {X_test.shape[0]} samples")
    
    # 4. Initialize and Train the Random Forest Model
    print("\n[Training] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model training complete.")
    
    # 5. Evaluate the model on unseen test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n==========================================")
    print("         MODEL PERFORMANCE REPORT         ")
    print("==========================================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On-Time (0)", "Delayed (1)"]))
    print("==========================================")
    
    # 6. Save the trained model and encoders to local files
    print("\n[Save] Saving trained model and label encoders to disk...")
    joblib.dump(model, "delay_predictor_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    print("✓ Saved model as 'delay_predictor_model.pkl'")
    print("✓ Saved encoders as 'label_encoders.pkl'")

if __name__ == "__main__":
    train_model()