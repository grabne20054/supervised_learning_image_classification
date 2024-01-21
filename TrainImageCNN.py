import os
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

class TrainImageCNN:
    def __init__(self):
        pass
    dataset_path = {
    "train": "train",
    }

    def CreateModel(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(len(self.GenerateTraining().class_indices), activation='softmax'))

        return model
    
    def CompileModel(self):
        model = self.CreateModel()
        model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return model

    def TrainModel(self, epochs=10):
        model = self.CompileModel()
        model.fit(
        self.GenerateTraining(),
        steps_per_epoch=self.GenerateTraining().samples // 32,
        epochs=epochs
        )

    def SaveModel(self, name="ImageClassification"):
        model = self.CreateModel()
        model.save(f"{name}.keras")

    def LoadModel(self, model_path):
        return models.load_model(model_path)

    def GenerateTraining(self, batch_size=32, image_size=(224,224)):
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path["train"],
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
        )
        return train_generator
