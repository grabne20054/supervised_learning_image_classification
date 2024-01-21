from TrainImageCNN import *

trainer = TrainImageCNN()
trainer.TrainModel(epochs=1000)
trainer.SaveModel("models/ImageClassification_0")