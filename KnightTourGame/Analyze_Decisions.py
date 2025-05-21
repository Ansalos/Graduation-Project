import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

# 1. Load the decision log
df = pd.read_csv("puct_decisions.csv")

# 2. Basic stats
print(f"Total decisions logged: {len(df)}")
print(df.head())

# 3. Plot correlation matrix
import seaborn as sns
corr = df.drop(columns=["step", "current_x", "current_y", "move_x", "move_y"]).corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. Train a decision tree classifier to predict the chosen move
features = ["degree", "Q", "P", "N", "PUCT"]
X = df[features]
y = df["is_best"]

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X, y)

# 5. Print human-readable rules
print("\nExtracted Heuristics (Decision Tree):")
print(export_text(clf, feature_names=features))

# 6. Optional: Accuracy
print("\nTraining Accuracy:", round(clf.score(X, y), 3))
