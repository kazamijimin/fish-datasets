# Fish Weight NLP Prediction Model
# This model uses Natural Language Processing to extract fish dimensions from text
# and predicts the weight using a trained Linear Regression model

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PART 1: LOAD AND PREPARE DATA
# =============================================================================

def load_and_train_model(csv_path='Fish[1].csv'):
    """Load fish data and train the linear regression model."""
    
    # Load the dataset
    df_fish = pd.read_csv(csv_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df_fish.shape}")
    print(f"Columns: {list(df_fish.columns)}")
    
    # Define features and target
    X = df_fish[['Length3', 'Height', 'Width']]
    y = df_fish['Weight']
    
    # Handle missing values
    X = X.dropna()
    y = y.loc[X.index]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model info
    print(f"\nModel Trained Successfully!")
    print(f"Intercept: {model.intercept_:.2f}")
    for i, col in enumerate(X_train.columns):
        print(f"Coefficient for {col}: {model.coef_[i]:.2f}")
    
    # Calculate R² score
    r2_score = model.score(X_test, y_test)
    print(f"R² Score: {r2_score:.4f}")
    
    return model, X_train, X_test, y_train, y_test

# =============================================================================
# PART 2: UNIT CONVERSION
# =============================================================================

UNIT_TO_CM = {
    "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
    "cm": 1.0, "centimeter": 1.0, "centimeters": 1.0,
    "m": 100.0, "meter": 100.0, "meters": 100.0,
    "in": 2.54, "inch": 2.54, "inches": 2.54, '"': 2.54,
    "ft": 30.48, "foot": 30.48, "feet": 30.48, "'": 30.48,
}

def to_cm(value: float, unit: str) -> float:
    """Convert a value from any supported unit to centimeters."""
    unit = unit.lower().strip()
    if unit not in UNIT_TO_CM:
        raise ValueError(f"Unsupported unit: {unit}")
    return float(value) * UNIT_TO_CM[unit]

def normalize_text(text: str) -> str:
    """Normalize text for easier parsing."""
    text = text.strip()
    
    # Make "5m" -> "5 m", "2mm" -> "2 mm", "2in" -> "2 in"
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet|foot)\b',
                  r'\1 \2', text, flags=re.IGNORECASE)
    
    # Normalize quote variants
    text = text.replace("'", "'").replace("″", '"').replace(""", '"').replace(""", '"')
    return text

# =============================================================================
# PART 3: NLP DIMENSION EXTRACTION
# =============================================================================

def extract_dimensions_any_unit(text: str):
    """
    Extract length, height, and width from natural language text.
    Supports various formats:
    - 'length: 25 cm, height: 8 cm, width: 4 cm'
    - '3 cm long, 20 cm height, 5 cm wide'
    - '25x8x4 cm'
    - '5m long 25cm height 2in wide'
    - '30, 12, 5' or '30 12 5' (defaults to cm)
    """
    text = normalize_text(text.lower())
    
    # Quick check: if just 3 numbers separated by commas/spaces, treat as L x H x W in cm
    simple_match = re.match(r'^\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*$', text.strip())
    if simple_match:
        return {
            "length_cm": float(simple_match.group(1)),
            "height_cm": float(simple_match.group(2)),
            "width_cm": float(simple_match.group(3))
        }
    
    # Keywords for each dimension
    length_keys = r"(?:length|long|len|l\b)"
    height_keys = r"(?:height|high|tall|ht|h\b)"
    width_keys  = r"(?:width|wide|wd|w\b)"
    
    # Unit and number patterns
    unit_group = r"(?:mm|cm|m|in|inch|inches|ft|foot|feet|\'|\")"
    num_group  = r"(?P<val>\d+(?:\.\d+)?)"
    unit_named = r"(?P<unit>" + unit_group + r")"
    
    patterns = {
        "length": [
            rf"{length_keys}\s*[:=]?\s*{num_group}\s*{unit_named}",
            rf"{num_group}\s*{unit_named}\s*{length_keys}",
        ],
        "height": [
            rf"{height_keys}\s*[:=]?\s*{num_group}\s*{unit_named}",
            rf"{num_group}\s*{unit_named}\s*{height_keys}",
        ],
        "width": [
            rf"{width_keys}\s*[:=]?\s*{num_group}\s*{unit_named}",
            rf"{num_group}\s*{unit_named}\s*{width_keys}",
        ],
    }
    
    def find_first_cm(key: str):
        for pat in patterns[key]:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return to_cm(float(m.group("val")), m.group("unit"))
        return None
    
    length_cm = find_first_cm("length")
    height_cm = find_first_cm("height")
    width_cm  = find_first_cm("width")
    
    # Fallback: "25x8x4 cm" => (Length, Height, Width)
    if length_cm is None or height_cm is None or width_cm is None:
        m = re.search(
            rf"(?P<v1>\d+(?:\.\d+)?)\s*[x×]\s*(?P<v2>\d+(?:\.\d+)?)\s*[x×]\s*(?P<v3>\d+(?:\.\d+)?)\s*(?P<u>{unit_group})\b",
            text, flags=re.IGNORECASE
        )
        if m:
            u = m.group("u")
            l = to_cm(float(m.group("v1")), u)
            h = to_cm(float(m.group("v2")), u)
            w = to_cm(float(m.group("v3")), u)
            length_cm = l if length_cm is None else length_cm
            height_cm = h if height_cm is None else height_cm
            width_cm  = w if width_cm  is None else width_cm
    
    return {"length_cm": length_cm, "height_cm": height_cm, "width_cm": width_cm}

def ask_value_with_unit(name: str) -> float:
    """Ask user for a value with unit if not detected. Defaults to cm if no unit given."""
    raw = normalize_text(input(f"{name} (e.g., 25 cm, 0.3 m, 2 in) [default: cm]: ").strip().lower())
    
    # Try to match with unit
    m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|inches|ft|foot|feet|\'|\")\b", raw)
    if m:
        return to_cm(float(m.group("val")), m.group("unit"))
    
    # Try to match just a number (default to cm)
    m_num = re.search(r"(?P<val>\d+(?:\.\d+)?)", raw)
    if m_num:
        print(f"  (No unit specified, using cm)")
        return float(m_num.group("val"))
    
    raise ValueError("Invalid format. Please enter a number.")

# =============================================================================
# PART 4: PREDICTION FUNCTION
# =============================================================================

def predict_fish_weight(model, X_train, user_input=None):
    """
    Main prediction function.
    - Takes natural language input
    - Extracts dimensions using NLP
    - Converts to cm
    - Checks if values are within training range
    - Predicts weight
    """
    
    # Get user input
    if user_input is None:
        user_input = input(
            "\nEnter fish dimensions with units. Examples:\n"
            "- '3 cm long, 20 cm height, 5 cm width'\n"
            "- '5m long 25cm height 2in wide'\n"
            "- '25x8x4 cm'\n"
            "- 'length: 30 cm, height: 12 cm, width: 5 cm'\n> "
        )
    
    # Extract dimensions
    dims = extract_dimensions_any_unit(user_input)
    print("\nExtracted (converted to cm):", dims)
    
    length = dims["length_cm"]
    height = dims["height_cm"]
    width  = dims["width_cm"]
    
    # Ask for missing values
    if length is None:
        print("\nLength not detected.")
        length = ask_value_with_unit("Length")
    if height is None:
        print("\nHeight not detected.")
        height = ask_value_with_unit("Height")
    if width is None:
        print("\nWidth not detected.")
        width = ask_value_with_unit("Width")
    
    # Build input DataFrame
    feature_cols = list(X_train.columns)
    row = {}
    for col in feature_cols:
        c = col.lower()
        if c == "length3":
            row[col] = length
        elif c == "height":
            row[col] = height
        elif c == "width":
            row[col] = width
        else:
            row[col] = 0.0
    
    X_input = pd.DataFrame([row], columns=feature_cols)
    print("\nInput used for prediction:\n", X_input)
    
    # Range warnings
    print("\n--- Range Check ---")
    for col in X_train.columns:
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        user_val = X_input[col].values[0]
        
        if user_val < min_val or user_val > max_val:
            print(f"Warning: {col} is outside training range ({min_val:.2f} - {max_val:.2f}). "
                  f"Your input = {user_val:.2f}. Prediction may be unreliable.")
        else:
            print(f"OK - {col}: {user_val:.2f} (within range {min_val:.2f} - {max_val:.2f})")
    
    # Predict
    pred = model.predict(X_input)
    pred_g = max(0.0, float(pred[0]))  # Prevent negative weights
    
    # Convert units
    pred_kg = pred_g / 1000.0
    pred_lb = pred_g / 453.592
    
    # Output results
    print(f"\n{'='*50}")
    print(f"PREDICTED WEIGHT")
    print(f"{'='*50}")
    print(f"  {pred_g:.2f} grams")
    print(f"  {pred_kg:.4f} kg")
    print(f"  {pred_lb:.4f} lb")
    print(f"{'='*50}")
    
    return X_input, pred_g

# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def visualize_prediction(model, X_train, y_train, X_input, feature_name="Length3"):
    """Create a visualization showing the prediction on the regression line."""
    
    # Training data
    x_train = X_train[feature_name].values
    y_train_vals = y_train.values
    
    # Fix other features at their mean
    fixed_vals = {c: X_train[c].mean() for c in X_train.columns if c != feature_name}
    
    # Create smooth x values
    x_line = np.linspace(x_train.min(), x_train.max(), 200)
    
    # Build prediction dataframe for the line
    line_df = pd.DataFrame({feature_name: x_line})
    for c, v in fixed_vals.items():
        line_df[c] = v
    
    # Predict line
    y_line = model.predict(line_df)
    
    # Predicted user point
    pred_x = float(X_input[feature_name].values[0])
    pred_y = float(model.predict(X_input)[0])
    
    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(x_train, y_train_vals, alpha=0.5, label="Training Data", color='blue')
    plt.plot(x_line, y_line, linewidth=2, color='red', label=f"Regression line (others fixed at mean)")
    plt.scatter(pred_x, pred_y, s=200, marker="X", color='green', edgecolors='black', 
                linewidths=2, label=f"Your Fish ({pred_y:.0f}g)")
    plt.xlabel(f"{feature_name} (cm)", fontsize=12)
    plt.ylabel("Weight (grams)", fontsize=12)
    plt.title("Fish Weight Prediction using NLP + Linear Regression", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return pred_x, pred_y

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   FISH WEIGHT NLP PREDICTION MODEL")
    print("="*60)
    
    # Load data and train model
    model, X_train, X_test, y_train, y_test = load_and_train_model('Fish[1].csv')
    
    # Run prediction loop
    while True:
        print("\n" + "-"*60)
        X_input, predicted_weight = predict_fish_weight(model, X_train)
        
        # Visualize
        visualize_prediction(model, X_train, y_train, X_input)
        
        # Ask to continue
        again = input("\nPredict another fish? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\nThank you for using the Fish Weight Predictor!")
            break
