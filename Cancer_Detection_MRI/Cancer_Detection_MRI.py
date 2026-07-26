# ==========================================================
# Project: Cancer Detection using MRI Images
# Author: Abhishek Thakur
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 50)
    print("Loading MRI Dataset...")
    print("=" * 50)

    train_dataset = image_dataset_from_directory(

        TRAIN_DIR,

        image_size=(128, 128),

        batch_size=32,

        shuffle=True

    )

    test_dataset = image_dataset_from_directory(

        TEST_DIR,

        image_size=(128, 128),

        batch_size=32,

        shuffle=False

    )

    class_names = train_dataset.class_names

    print("\nClasses :", class_names)

    return train_dataset, test_dataset, class_names


# ==========================================================
# Normalize Images
# ==========================================================

def normalize(train_dataset, test_dataset):

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    return train_dataset, test_dataset


# ==========================================================
# Display Sample Images
# ==========================================================

def display_images(train_dataset, class_names):

    plt.figure(figsize=(10,10))

    for images, labels in train_dataset.take(1):

        for i in range(9):

            plt.subplot(3,3,i+1)

            plt.imshow(images[i].numpy().astype("uint8"))

            plt.title(class_names[labels[i]])

            plt.axis("off")

    plt.tight_layout()

    plt.show()

# ==========================================================
# Build CNN Model
# ==========================================================

def build_model(num_classes):

    print("\nBuilding CNN Model...")

    model = Sequential([

        Input(shape=(128, 128, 3)),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.5),

        Dense(num_classes, activation="softmax")

    ])

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
# Train Model
# ==========================================================

def train_model(model, train_dataset, test_dataset):

    print("\nTraining Model...\n")

    history = model.fit(

        train_dataset,

        validation_data=test_dataset,

        epochs=10,

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

def evaluate_model(model, test_dataset):

    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(test_dataset, verbose=1)

    print("\nTest Accuracy : {:.2f}%".format(accuracy * 100))
    print("Test Loss     : {:.4f}".format(loss))


# ==========================================================
# Predict Test Images
# ==========================================================

def predict_images(model, test_dataset):

    print("\nPredicting MRI Images...\n")

    predictions = model.predict(test_dataset)

    return predictions


# ==========================================================
# Display Prediction Results
# ==========================================================

def display_predictions(predictions, test_dataset, class_names):

    plt.figure(figsize=(12, 8))

    images, labels = next(iter(test_dataset))

    for i in range(9):

        plt.subplot(3,3,i+1)

        plt.imshow(images[i].numpy().astype("uint8"))

        predicted = np.argmax(predictions[i])

        actual = labels[i].numpy()

        plt.title(
            f"P: {class_names[predicted]}\nA: {class_names[actual]}",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Predict a Single MRI Image
# ==========================================================

def predict_single_image(model, test_dataset, class_names):

    images, labels = next(iter(test_dataset))

    image = np.expand_dims(images[0], axis=0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    actual_class = labels[0].numpy()

    print("\nSingle MRI Prediction")
    print("-" * 40)

    print("Actual Class    :", class_names[actual_class])
    print("Predicted Class :", class_names[predicted_class])

    plt.figure(figsize=(4,4))

    plt.imshow(images[0].numpy().astype("uint8"))

    plt.title(f"Prediction: {class_names[predicted_class]}")

    plt.axis("off")

    plt.show()


# ==========================================================
# Save Model
# ==========================================================

def save_model(model):

    print("\nSaving Model...")

    model.save("brain_tumor_cnn.keras")

    print("Model saved successfully as 'brain_tumor_cnn.keras'")

# ==========================================================
# Main Function
# ==========================================================

def main():

    # Load Dataset
    train_dataset, test_dataset, class_names = load_dataset()

    # Normalize Images
    train_dataset, test_dataset = normalize(
        train_dataset,
        test_dataset
    )

    # Display Sample Images
    display_images(train_dataset, class_names)

    # Build CNN Model
    model = build_model(len(class_names))

    # Compile Model
    compile_model(model)

    # Train Model
    history = train_model(
        model,
        train_dataset,
        test_dataset
    )

    # Plot Accuracy
    plot_accuracy(history)

    # Plot Loss
    plot_loss(history)

    # Evaluate Model
    evaluate_model(
        model,
        test_dataset
    )

    # Predict Images
    predictions = predict_images(
        model,
        test_dataset
    )

    # Display Predictions
    display_predictions(
        predictions,
        test_dataset,
        class_names
    )

    # Predict Single MRI Image
    predict_single_image(
        model,
        test_dataset,
        class_names
    )

    # Save Model
    save_model(model)

    print("\n" + "=" * 60)
    print("Cancer Detection using MRI Images Completed Successfully!")
    print("=" * 60)


# ==========================================================
# Execute Program
# ==========================================================

if __name__ == "__main__":
    main()