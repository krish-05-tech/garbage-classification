from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model.h5")
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))  # match your model's input size
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Predicted class:", np.argmax(pred))
