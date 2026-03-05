# Step 3: Perform Linear Regression and Plot the Regression Graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df_fish = pd.read_csv('Fish[1].csv')

print("=" * 60)
print("LINEAR REGRESSION MODEL")
print("=" * 60)

# Define features (X) and target (y) for multivariable regression
# Using 'Length3', 'Height', 'Width' as independent variables
# and 'Weight' as the dependent variable
X_multi = df_fish[['Length3', 'Height', 'Width']]
y = df_fish['Weight']

print("\nFeatures (X):", list(X_multi.columns))
print("Target (y): Weight")

# Handle potential missing values
X_multi = X_multi.dropna()
y = y.loc[X_multi.index]

print(f"\nDataset size: {len(X_multi)} samples")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Initialize and train the Multivariable Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\n" + "-" * 60)
print("MODEL COEFFICIENTS")
print("-" * 60)
print(f"Intercept: {model.intercept_:.2f}")
for i, col in enumerate(X_train.columns):
    print(f"Coefficient for {col}: {model.coef_[i]:.2f}")

print("\n" + "-" * 60)
print("MODEL PERFORMANCE METRICS")
print("-" * 60)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} g")
print(f"Mean Absolute Error (MAE): {mae:.2f} g")

print("\n" + "-" * 60)
print("TRAINING DATA RANGES")
print("-" * 60)
for col in X_train.columns:
    print(f"{col}: {X_train[col].min():.2f} - {X_train[col].max():.2f}")

# Create visualization: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Weight (g)', fontsize=12)
ax1.set_ylabel('Predicted Weight (g)', fontsize=12)
ax1.set_title('Actual vs. Predicted Weights\n(Multivariable Linear Regression)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add R² annotation
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Residuals
residuals = y_test - y_pred
ax2 = axes[1]
ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Weight (g)', fontsize=12)
ax2.set_ylabel('Residuals (g)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=150)
print("\nRegression plots saved as 'regression_results.png'")
plt.show()

# Create 2D slice visualization
fig2, ax3 = plt.subplots(figsize=(10, 7))

feature_name = "Length3"
x_train_vals = X_train[feature_name].values
y_train_vals = y_train.values

# Fix other features at their mean
fixed_vals = {c: X_train[c].mean() for c in X_train.columns if c != feature_name}

# Create smooth x values for the regression line
x_line = np.linspace(x_train_vals.min(), x_train_vals.max(), 200)

# Build prediction dataframe for the line
line_df = pd.DataFrame({feature_name: x_line})
for c, v in fixed_vals.items():
    line_df[c] = v

# Predict line
y_line = model.predict(line_df)

# Plot
ax3.scatter(x_train_vals, y_train_vals, alpha=0.5, label="Training Data", color='blue')
ax3.plot(x_line, y_line, linewidth=2, color='red', 
         label=f"Regression line (Height={fixed_vals['Height']:.1f}, Width={fixed_vals['Width']:.1f})")
ax3.set_xlabel(f"{feature_name} (cm)", fontsize=12)
ax3.set_ylabel("Weight (grams)", fontsize=12)
ax3.set_title("Multivariable Linear Regression (2D Slice)", fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig('regression_2d_slice.png', dpi=150)
print("2D slice plot saved as 'regression_2d_slice.png'")
plt.show()

print("\n" + "=" * 60)
print("Linear Regression complete!")
print("=" * 60)
