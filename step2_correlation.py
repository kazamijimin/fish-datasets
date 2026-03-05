# Step 2: Calculate and Visualize the Correlation Matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df_fish = pd.read_csv('Fish[1].csv')

print("=" * 60)
print("CORRELATION MATRIX ANALYSIS")
print("=" * 60)

# Select only numerical columns for correlation calculation
numerical_df_fish = df_fish.select_dtypes(include=['number'])

print("\nNumerical columns used:")
for col in numerical_df_fish.columns:
    print(f"  - {col}")

# Calculate the correlation matrix
correlation_matrix = numerical_df_fish.corr()

print("\n" + "-" * 60)
print("Correlation Matrix:")
print(correlation_matrix.round(2))

print("\n" + "-" * 60)
print("Key Observations:")
print("  - Values close to 1: Strong positive correlation")
print("  - Values close to -1: Strong negative correlation")
print("  - Values close to 0: Weak or no correlation")

# Find strongest correlations with Weight
weight_corr = correlation_matrix['Weight'].drop('Weight').sort_values(ascending=False)
print("\n" + "-" * 60)
print("Correlations with Weight (target variable):")
for feature, corr in weight_corr.items():
    print(f"  {feature}: {corr:.3f}")

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Fish Dataset Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
print("\nCorrelation heatmap saved as 'correlation_matrix.png'")
plt.show()

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
