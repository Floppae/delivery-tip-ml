import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/synthetic/synthetic_delivery_data.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df['tip_percent'].describe())

# Tip percent distribution
df['tip_percent'].hist(bins=30)
plt.title("Tip Percent Distribution")
plt.xlabel("Tip Percent")
plt.ylabel("Frequency")
plt.show()

# Scatter plots for numeric features
numeric_features = [
    "distance_miles",
    "wait_time_minutes",
    "order_subtotal",
    "item_count",
    "messages_sent"
]

for col in numeric_features:
    plt.scatter(df[col], df['tip_percent'])
    plt.title(f"{col} vs tip_percent")
    plt.xlabel(col)
    plt.ylabel("tip_percent")
    plt.show()

# Boxplots for categorical features
categorical_features = ["weather", "time_of_day", "day_of_week"]

for col in categorical_features:
    df.boxplot(column="tip_percent", by=col)
    plt.title(f"tip_percent by {col}")
    plt.suptitle("")
    plt.show()

# Correlation heatmap
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
