import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the decision log
df = pd.read_csv("puct_decisions.csv")

# 2. Strip spaces from column names (just in case)
df.columns = df.columns.str.strip()

# 3. Check if DataFrame is empty
if df.empty:
    print("CSV file is empty or failed to load.")
    exit()

# 4. Print columns to verify
print("Columns in the CSV file:")
print(df.columns.tolist())

# 5. Basic stats
print(f"\nTotal decisions logged: {len(df)}")
print(df.head())

# 6. Plot correlation matrix
drop_cols = ["step", "current_x", "current_y", "move_x", "move_y"]
drop_cols = [col for col in drop_cols if col in df.columns]
corr = df.drop(columns=drop_cols).corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 7. Train a decision tree classifier
features = ["degree", "Q", "P", "N", "PUCT"]
for feature in features:
    if feature not in df.columns:
        raise KeyError(f"Missing expected feature column: {feature}")

X = df[features]
y = df["is_best"]

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# 8. Print human-readable rules
print("\nExtracted Heuristics (Decision Tree):")
print(export_text(clf, feature_names=features))

# 9. Optional: Training accuracy
print("\nTraining Accuracy:", round(clf.score(X, y), 3))

# 10. Plot the decision tree
plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=features, class_names=["Not Chosen", "Best Move"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Explaining Move Selection (PUCT-based AI)")
plt.show()
