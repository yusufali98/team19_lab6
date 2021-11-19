import tensorflow
from PIL import Image, ImageOps
import numpy as np
import tqdm
import os

test_data_folder = '/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/test_images/'

model = tensorflow.keras.models.load_model('/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/keras_model.h5', compile=False)

print(os.listdir(test_data_folder))

idx = 1

pred_list = []

for files in (os.listdir(test_data_folder)):

    fname = test_data_folder + files

    print(fname)

    data = np.ndarray(shape = (1,224,224,3), dtype = np.float32)

    image = Image.open(fname)

    size = (224,224)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_arr = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_arr

    prediction = model.predict(data)

    prediction_idx = np.argmax(prediction) + 1

    if prediction_idx == 6:
        prediction_idx = 0

    print("\n Prediction ", fname, " : ", prediction_idx)

    pred_list.append(prediction_idx)

    idx += 1

print(pred_list)

pred_list = np.asarray(pred_list)

print(pred_list)

