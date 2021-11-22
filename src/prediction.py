import tensorflow
from PIL import Image, ImageOps
import numpy as np
import tqdm
import os
import glob


def evaluate():

    test_data_folder = '../test_images/'

    model = tensorflow.keras.models.load_model('../keras_model.h5', compile=False)

    idx = 1

    pred_list = []
    file_names = []

    os.chdir(test_data_folder)

    for file in glob.glob("*.jpg"):
        file_names.append(file)

    file_names = [x[:-4] for x in file_names]
    file_names = [int(x) for x in file_names]
    file_names.sort()

    # print(file_names)

    file_names = [str(x) + '.jpg' for x in file_names]
    # print(file_names)

    for files in file_names:

        fname = test_data_folder + files

        # print(fname)

        data = np.ndarray(shape = (1,224,224,3), dtype = np.float32)

        image = Image.open(fname)

        size = (224,224)

        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_arr = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_arr

        prediction = model.predict(data)

        prediction_idx = np.argmax(prediction) # + 1

        # if prediction_idx == 6:
        #     prediction_idx = 0

        # print("\n Prediction ", fname, " : ", prediction_idx)

        pred_list.append(prediction_idx)

        idx += 1

    # print(pred_list)

    pred_array = np.asarray(pred_list)

    # print("\n pred_list : \n", pred_array)
    # print("\n pred_list shape: ", pred_array.shape)

    test_labels_path = '../test_images/test.txt'

    test_labels = []

    with open(test_labels_path, "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            label = currentline[1]
            test_labels.append(label)

    test_labels = [x[:-1] for x in test_labels]

    labels_array = np.asarray(test_labels)

    # print("\n labels_list : \n", labels_array)
    # print("\n labels_list shape: ", labels_array.shape)

    pred_array = pred_array.astype(int)
    labels_array = labels_array.astype(int)

    pred_array = tensorflow.convert_to_tensor(pred_array)
    labels_array = tensorflow.convert_to_tensor(labels_array)

    confusion_matrix = tensorflow.math.confusion_matrix(labels_array, pred_array)

    print("\n Confusion Matrix: \n")
    tensorflow.print(confusion_matrix)

    m = tensorflow.keras.metrics.Accuracy()

    m.update_state(y_true = labels_array, y_pred = pred_array)

    accuracy = m.result()

    print("\n Accuracy: \n")
    tensorflow.print(accuracy)


if __name__ == "__main__":
    evaluate()