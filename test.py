import os
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from TrainImageCNN import * 

dataset_path = {
    "test": "test"}

batch_size=32
image_size=(224,224)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_path["test"],
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle the test data
)


test_image_path = "test/volleyball/1.jpg"  # Replace with an actual image path
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=image_size)
test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_image_array = tf.expand_dims(test_image_array, 0)  # Add batch dimension
test_image_array /= 255.0  # Normalize pixel values to between 0 and 1

# Make a prediction
model_path = "models/ImageClassification_0.keras"

trainer = TrainImageCNN()
model = trainer.CreateModel()
trainer.CompileModel()
loadedmodel = trainer.LoadModel(model_path)
predictions = loadedmodel.predict(test_image_array)
predicted_class = tf.argmax(predictions[0]).numpy()

# Print the result
print("Predicted class index:", predicted_class)
print("Predicted class:", list(trainer.GenerateTraining().class_indices.keys())[predicted_class])
