from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
iris = load_iris(as_frame=True)
# 2. Combine features and target into a single DataFrame
df = iris.frame
# 3. Save the DataFrame as a CSV file in your project folder
df.to_csv("iris.csv", index=False)
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "iris_model.pkl")

