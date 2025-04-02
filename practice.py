#scikit learn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import numpy as np
data = load_breast_cancer()

x = data.data
y = data.target

scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=42)

print(np.shape(X_train))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs= 50, batch_size=16,validation_split=0.1, verbose=0)

loss, accuracy = model.evaluate(X_test, Y_test)

print("\nTest Accuracy:", accuracy)

preds = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(Y_test, preds))
