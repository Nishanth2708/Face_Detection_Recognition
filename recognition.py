import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time

a= 'data/'
subjects = [ '', "Nishanth Vanipenta"]
faces =[]
labels =[]
x_test =[]
print('showing faces.....\n')

for imgfolder in os.listdir(a):
    # print(imgfolder)
    for filename in os.listdir(a + imgfolder):
        filename = a + imgfolder + '/' + filename
        img = cv2.imread(filename,0)
        cv2.imshow('k',img)
        cv2.waitKey(100)
        faces.append(img)

        # print(resize.shape)
#
for imgfolder in os.listdir(a):
    for filename in os.listdir(a + imgfolder):
        labels.append(imgfolder)

print(labels)
print('Total faces:', len(faces))
print('Total Labels',len(labels))


recognizer = cv2.face.LBPHFaceRecognizer_create()
training = recognizer.train(faces,np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


test_img = r'test\Nishanth\WhatsApp Image 2019-10-30 at 11.03.21 PM.jpeg'
label = recognizer.predict(test_img)
text =  subjects[label]
draw_text(test_img, text, 50, 45)


# for imgfolder in os.listdir('test/'):
#     for filename in os.listdir('test/' +imgfolder):
#         if (filename.endswith('.jpeg')):
#             filename = 'test/' + imgfolder + '/' + filename
#             # print(filename)
#             img = cv2.imread(filename, 0)
#             img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
#             # print('x_test',img.shape)
#             x_test.append(img)
#
# y_test = []
# for imgfolder in os.listdir('test/'):
#     for filename in os.listdir('test/' + imgfolder):
#         y_test.append(imgfolder)
#
# x_imgs = np.asarray(x_imgs)
# y_train = np.asarray(y_train)
# y_test = np.asarray(y_test)
# x_test = np.asarray(x_test)
#
#
# fig,ax = plt.subplots(3,6)
# for i, axis in enumerate(ax.flat):
#     axis.imshow(x_imgs[i], cmap= 'gray')
#     axis.set(xticks = [], yticks=[], xlabel=y_train[i])
#
# fig, ax = plt.subplots(2, 2)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(x_test[i], cmap='gray')
#     axi.set(xticks=[], yticks=[],
#             xlabel=y_test[i])
# # plt.show()
# X_data = x_imgs.reshape(x_imgs.shape[0], x_imgs.shape[1] * x_imgs.shape[2])
# X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
#












# print(x_imgs.shape)
# for filename in glob.glob(path):
#     if filename.endswith('.jpg'):
#         imgs.append(filename)
#
# x_images =[]
# for img in imgs:
#     # print(img)
#     each_img = cv2.imread(img)
#     gray_img=  cv2.cvtColor(each_img,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('img',gray_img)
#     cv2.waitKey(0)
#     each_img = cv2.resize(each_img, (47,62), interpolation = cv2.INTER_AREA)
#     # print(each_img.shape)
#     x_images.append(each_img)
#
# x_images = np.asarray(x_images)
# # print(x_images.shape)
#
# cv2.imshow('img',x_images)
# cv2.waitKey(0)