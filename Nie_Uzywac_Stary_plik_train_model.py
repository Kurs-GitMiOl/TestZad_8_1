# Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from pathlib import Path

# See what the Iris data looks like
# Wczytanie zbioru Iris
iris = load_iris()

# Utworzenie DataFrame z Pandas
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Dodanie kolumny z nazwą gatunku
df['species'] = [iris['target_names'][i] for i in iris['target']]

# Podejrzenie pierwszych 5 wierszy
# Preview the first 5 rows
print("\nPreview the first 5 rows")
print(df.head(5))

# Basic statistics of the dataset
print("\nBasic statistics of the datase")
print(df.describe())

# Wczytujemy dane Iris
# Load Iris data (features and labels) as DataFrame
X, y = load_iris(return_X_y=True, as_frame=True)

# Dzielimy dane na trening i test
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tworzymy pipeline: preprocessing + model
# Pipeline: feature scaling + SVM classifier
model = Pipeline([
    ('scaler', StandardScaler()),   # skalowanie cech
    ('clf', SVC(kernel='linear', probability=True))   # klasyfikator SVM
])

# z przykładową redukcją cech nie jest tu konieczna bo  sa tylko 4 cechy
# model = Pipeline([
#     ('scaler', StandardScaler()),   # skalowanie cech
#     ('pca', PCA(n_components=3)),   # redukcja wymiarów
#     ('clf', SVC(kernel='linear', probability=True))   # klasyfikator SVM
# ])


# Trenujemy pipeline
# Train the pipeline on the training data
model.fit(X_train, y_train)


# Sprawdzamy skuteczność
# Check the model accuracy
print("\nAccuracy:", model.score(X_test, y_test))

# Zapisujemy CAŁY pipeline do pliku
# Save the trained pipeline to a file
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/iris_model.joblib")

print("Model saved to model directory model/iris_model.joblib")