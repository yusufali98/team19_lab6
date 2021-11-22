import cv2
import os
'''
left_data_folder = '../left_train_imgs/'

file_names = os.listdir(left_data_folder)

print(file_names)

for i,files in enumerate(file_names):

    fname = left_data_folder + files

    # print(fname)

    img = cv2.imread(fname)

    print(img.shape)

    img_flip_lr = cv2.flip(img, 1)

    cv2.imshow('flipped', img_flip_lr)
    cv2.waitKey(0)

    cv2.imwrite('/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/left_flipped/' + str(i) + 'flip.jpg', img_flip_lr)
'''
# -----------------------------------------------------------------------------------------------------------------------------------

left_data_folder = '../right_train_imgs/'

file_names = os.listdir(left_data_folder)

print(file_names)

for i,files in enumerate(file_names):

    fname = left_data_folder + files

    # print(fname)

    img = cv2.imread(fname)

    print(img.shape)

    img_flip_lr = cv2.flip(img, 1)

    cv2.imshow('flipped', img_flip_lr)
    cv2.waitKey(0)

    cv2.imwrite('/home/yusuf/Desktop/CS7785/Lab_6/lab6_ws/src/team19_lab6/right_flipped/' + str(i) + 'flip.jpg', img_flip_lr)