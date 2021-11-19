import tensorflow
from PIL import Image, ImageOps
import numpy as np

model = tensorflow.keras.models.load_model('/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/keras_model.h5', compile=False)

data = np.ndarray(shape = (1,224,224,3), dtype = np.float32)

image = Image.open('/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/test_images/10.jpg')

size = (224,224)

image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)

normalized_image_arr = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_arr

prediction = model.predict(data)

print("\n Prediction: ", prediction)