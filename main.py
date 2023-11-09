import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class Visualization:
    def __init__(self) -> None:
        pass
    def visualize_history(self, history): 
        plt.figure(figsize=(10, 16))
        plt.rcParams['figure.figsize'] = [16, 9]
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.grid'] = True
        plt.rcParams['figure.facecolor'] = 'white'

        # Accuracy
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.title(f'\nTraining and Validation Accuracy.\nTrain Accuracy: {str(history.history["accuracy"][-1])}\nValidation Accuracy: {str(history.history["val_accuracy"][-1])}')

        # Loss
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title(f'Training and Validation Loss.\nTrain Loss: {str(history.history["loss"][-1])}\nValidation Loss: {str(history.history["val_loss"][-1])}')
        plt.xlabel('epoch')
        plt.tight_layout(pad=3.0)
        
class Classification:
    def __init__(self,train_path,test_path,labels,sample):
        self.visualizer = Visualization()
        self.train_path = train_path
        self.test_path = test_path
        self.labels = labels
        self.sample = sample
    

        # Update file extensions
        self.labels['id'] = self.labels['id'].apply(lambda id: id + '.jpg')
        self.sample['id'] = self.sample['id'].apply(lambda id: id + '.jpg')

        # Data augmentation and preprocessing
        self.gen = ImageDataGenerator(
            rescale=1./255.,
            horizontal_flip=True,
            validation_split=0.2
        )

    def create_data_generator(self, data_frame, subset):
        return self.gen.flow_from_dataframe(
            data_frame,
            directory=train_path,
            x_col='id',
            y_col='breed',
            subset=subset,
            color_mode="rgb",
            target_size=(331, 331),
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
            seed=20
        )
    
    def classify(self):
        train_generator = self.create_data_generator(self.labels, "training")
        validation_generator = self.create_data_generator(self.labels, "validation")

        # Build the model
        base_model = self.create_base_model()
        model = self.create_model(base_model)

        # Learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=10000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping callback
        early = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

        # Define batch size and steps per epoch
        batch_size = 32
        STEP_SIZE_TRAIN = train_generator.n // batch_size
        STEP_SIZE_VALIDATION = validation_generator.n // batch_size

        # Fit the model
        history = model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_VALIDATION,
            epochs=1,
            callbacks=[early]
        )

        model.save("Model.h5")

        self.visualizer.visualize_history(history)

        # Evaluate the model
        accuracy_score = model.evaluate(validation_generator)
        print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))
        print("Loss: ", accuracy_score[0])

        # Make predictions and create a submission file
        predictions = []
        for image in sample.id:
            print(image)
            img = tf.keras.preprocessing.image.load_img(test_path + '/' + image)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.preprocessing.image.smart_resize(img, (331, 331))
            img = tf.reshape(img, (-1, 331, 331, 3))
            prediction = model.predict(img / 255)
            # predictions.append(np.argmax(prediction))
            predictions.append(prediction)

        my_submission = pd.DataFrame({'image_id': sample.id, 'label': predictions})
        my_submission.to_csv('submission.csv', index=False)

        # Display the first five predicted outputs
        print("Submission File:\n---------------")
        print(my_submission.head())

    def create_base_model(self):
        base_model = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(331, 331, 3)
        )
        # Fine-tune the top layers of the base model
        # fine_tune_at = 150 # 164 layers  
        # for layer in base_model.layers[:fine_tune_at]:
        #     layer.trainable = False
        base_model.trainable = False
        return base_model

    def create_model(self, base_model):
        model = Sequential([
            base_model,
            BatchNormalization(renorm=True),
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(120, activation='softmax')
        ])
        
        return model

if __name__ == "__main__":
    train_path = "train"
    test_path = "test"

    # Load labels and sample data
    labels = pd.read_csv("labels.csv")
    sample = pd.read_csv('sample_submission.csv')

    classifier = Classification(train_path, test_path, labels, sample)
    classifier.classify()