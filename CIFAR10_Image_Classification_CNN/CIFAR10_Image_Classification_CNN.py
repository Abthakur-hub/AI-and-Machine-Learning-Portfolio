# ==========================================================
# Project: CIFAR-10 Image Classification using CNN
# Author: Abhishek Thakur
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():
    print("=" * 50)
    print("Loading CIFAR-10 Dataset...")
    print("=" * 50)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print("\nTraining Images :", X_train.shape)
    print("Training Labels :", y_train.shape)

    print("\nTesting Images :", X_test.shape)
    print("Testing Labels :", y_test.shape)

    return X_train, y_train, X_test, y_test


# ==========================================================
# Explore Dataset
# ==========================================================

def explore_dataset(X_train, y_train, X_test):

    print("\nDataset Information")
    print("-" * 40)

    print("Image Shape :", X_train[0].shape)

    print("Number of Classes :", len(np.unique(y_train)))

    print("Training Images :", len(X_train))

    print("Testing Images :", len(X_test))


# ==========================================================
# Define Class Names
# ==========================================================

class_names = [

    "Airplane",

    "Automobile",

    "Bird",

    "Cat",

    "Deer",

    "Dog",

    "Frog",

    "Horse",

    "Ship",

    "Truck"

]


# ==========================================================
# Display Sample Images
# ==========================================================

def display_images(X_train, y_train):

    plt.figure(figsize=(12,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(X_train[i])

        plt.title(class_names[y_train[i][0]], fontsize=8)

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Normalize Images
# ==========================================================

def normalize_images(X_train, X_test):

    X_train = X_train.astype("float32") / 255.0

    X_test = X_test.astype("float32") / 255.0

    print("\nImages Normalized Successfully.")

    return X_train, X_test
# ==========================================================
# Build CNN Model
# ==========================================================

def build_model():

    print("\nBuilding CNN Model...")

    model = Sequential([

        Input(shape=(32, 32, 3)),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),

        Flatten(),

        Dense(64, activation="relu"),

        Dropout(0.5),

        Dense(10, activation="softmax")

    ])

    print("\nCNN Model Summary\n")
    model.summary()

    return model


# ==========================================================
# Compile Model
# ==========================================================

def compile_model(model):

    print("\nCompiling Model...")

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("Model Compiled Successfully.")


# ==========================================================
# Train CNN Model
# ==========================================================

def train_model(model, X_train, y_train, X_test, y_test):

    print("\nTraining Started...\n")

    history = model.fit(

        X_train,

        y_train,

        epochs=10,

        batch_size=64,

        validation_data=(X_test, y_test),

        verbose=1

    )

    print("\nTraining Completed.")

    return history


# ==========================================================
# Plot Training Accuracy
# ==========================================================

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(

        history.history["accuracy"],

        label="Training Accuracy"

    )

    plt.plot(

        history.history["val_accuracy"],

        label="Validation Accuracy"

    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.show()


# ==========================================================
# Plot Training Loss
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(

        history.history["loss"],

        label="Training Loss"

    )

    plt.plot(

        history.history["val_loss"],

        label="Validation Loss"

    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("\nTest Accuracy : {:.2f}%".format(accuracy * 100))
    print("Test Loss     : {:.4f}".format(loss))


# ==========================================================
# Predict Test Images
# ==========================================================

def predict_images(model, X_test):

    print("\nPredicting Test Images...\n")

    predictions = model.predict(X_test)

    return predictions


# ==========================================================
# Display Prediction Results
# ==========================================================

def display_predictions(predictions, X_test, y_test):

    plt.figure(figsize=(12,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(X_test[i])

        predicted_label = np.argmax(predictions[i])

        actual_label = y_test[i][0]

        plt.title(
            f"P: {class_names[predicted_label]}\nA: {class_names[actual_label]}",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Predict Single Image
# ==========================================================

def predict_single_image(model, X_test, y_test):

    index = 25

    image = np.expand_dims(X_test[index], axis=0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    actual_class = y_test[index][0]

    print("\nSingle Image Prediction")
    print("-" * 35)

    print("Actual Class    :", class_names[actual_class])

    print("Predicted Class :", class_names[predicted_class])

    plt.figure(figsize=(4,4))

    plt.imshow(X_test[index])

    plt.title(f"Predicted: {class_names[predicted_class]}")

    plt.axis("off")

    plt.show()

# ==========================================================
# Save Trained Model
# ==========================================================

def save_model(model):

    print("\nSaving Model...")

    model.save("cifar10_cnn_model.keras")

    print("Model saved successfully as 'cifar10_cnn_model.keras'")


# ==========================================================
# Main Function
# ==========================================================

def main():

    # Load Dataset
    X_train, y_train, X_test, y_test = load_dataset()

    # Explore Dataset
    explore_dataset(X_train, y_train, X_test)

    # Display Sample Images
    display_images(X_train)

    # Normalize Images
    X_train, X_test = normalize_images(X_train, X_test)

    # Build CNN Model
    model = build_model()

    # Compile Model
    compile_model(model)

    # Train Model
    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Plot Accuracy
    plot_accuracy(history)

    # Plot Loss
    plot_loss(history)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Predict Images
    predictions = predict_images(model, X_test)

    # Display Predictions
    display_predictions(predictions, X_test, y_test)

    # Predict Single Image
    predict_single_image(model, X_test, y_test)

    # Save Model
    save_model(model)

    print("\n" + "=" * 60)
    print("CIFAR-10 Image Classification Project Completed Successfully!")
    print("=" * 60)


# ==========================================================
# Execute Program
# ==========================================================

if __name__ == "__main__":
    main()