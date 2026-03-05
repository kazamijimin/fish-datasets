# Step 1: Load the Fish[1].csv dataset and inspect its structure
import pandas as pd

# Load the dataset
df_fish = pd.read_csv('Fish[1].csv')

# Display the first 5 rows of the DataFrame
print("=" * 60)
print("FISH DATASET - STRUCTURE INSPECTION")
print("=" * 60)

print("\nFirst 5 rows of the Fish dataset:")
print(df_fish.head())

print("\n" + "-" * 60)
print("Dataset Shape:", df_fish.shape)
print(f"  - {df_fish.shape[0]} rows (fish samples)")
print(f"  - {df_fish.shape[1]} columns (features)")

print("\n" + "-" * 60)
print("Column Names:")
for i, col in enumerate(df_fish.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "-" * 60)
print("DataFrame Info:")
df_fish.info()

print("\n" + "-" * 60)
print("Statistical Summary:")
print(df_fish.describe())

print("\n" + "-" * 60)
print("Species Distribution:")
print(df_fish['Species'].value_counts())

print("\n" + "=" * 60)
print("Dataset loaded successfully!")
print("=" * 60)
