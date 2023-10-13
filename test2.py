import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    print("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Define data paths
train_path = "train"
test_path = "test"

# Load labels and sample data
labels = pd.read_csv("labels.csv")
sample = pd.read_csv("sample_submission.csv")

# Update file extensions
labels["id"] = labels["id"].apply(lambda id: id + ".jpg")
sample["id"] = sample["id"].apply(lambda id: id + ".jpg")

# Data augmentation and preprocessing
gen = ImageDataGenerator(
    rescale=1.0 / 255.0, horizontal_flip=True, validation_split=0.2
)


def create_data_generator(data_frame, subset):
    return gen.flow_from_dataframe(
        data_frame,
        directory=train_path,
        x_col="id",
        y_col="breed",
        subset=subset,
        color_mode="rgb",
        target_size=(331, 331),
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=42,
    )


train_generator = create_data_generator(labels, "training")
validation_generator = create_data_generator(labels, "validation")

# Build the model
base_model = InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=(331, 331, 3)
)
# Fine-tune the top layers of the base model
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
base_model.trainable = False

model = Sequential(
    [
        base_model,
        BatchNormalization(renorm=True),
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(120, activation="softmax"),
    ]
)

# Learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Early stopping callback
early = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

# Define batch size and steps per epoch
batch_size = 32
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=[early],
)

model.save("Model.h5")


# Plot results
def plot_history(history):
    plt.figure(figsize=(10, 16))
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.facecolor"] = "white"

    # Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.title(
        f'\nTraining and Validation Accuracy.\nTrain Accuracy: {str(history.history["accuracy"][-1])}\nValidation Accuracy: {str(history.history["val_accuracy"][-1])}'
    )

    # Loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.title(
        f'Training and Validation Loss.\nTrain Loss: {str(history.history["loss"][-1])}\nValidation Loss: {str(history.history["val_loss"][-1])}'
    )
    plt.xlabel("epoch")
    plt.tight_layout(pad=3.0)


plot_history(history)

# Evaluate the model
accuracy_score = model.evaluate(validation_generator)
print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))
print("Loss: ", accuracy_score[0])

# Test an image
test_img_path = test_path + "/000621fb3cbb32d8935728e48679680e.jpg"
img = cv2.imread(test_img_path)
resized_img = cv2.resize(img, (331, 331)).reshape(-1, 331, 331, 3) / 255

plt.figure(figsize=(6, 6))
plt.title("TEST IMAGE")
plt.imshow(resized_img[0])
plt.show()

# Make predictions and create a submission file
predictions = []
for image in sample.id:
    img = tf.keras.preprocessing.image.load_img(test_path + "/" + image)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (331, 331))
    img = tf.reshape(img, (-1, 331, 331, 3))
    prediction = model.predict(img / 255)
    predictions.append(np.argmax(prediction))

my_submission = pd.DataFrame({"image_id": sample.id, "label": predictions})
my_submission.to_csv("submission.csv", index=False)

# Display the first five predicted outputs
print("Submission File:\n---------------")
print(my_submission.head())
