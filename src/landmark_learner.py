import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


landmark_df: pd.DataFrame = pd.read_csv("./db/landmark_dataset.csv")
X = landmark_df.drop("label", axis=1).values
y = landmark_df["label"].values

std_sc = StandardScaler()
X_sc = std_sc.fit_transform(X)

clf = SVC(probability=True)
clf.fit(X, y)
y_pred = clf.predict(X)
print(classification_report(y, y_pred))

joblib.dump(clf, "./models/gesture_class_svm.pkl")
joblib.dump(std_sc, "./models/scaler.pkl")