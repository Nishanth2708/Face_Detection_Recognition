import numpy as np
import cv2
import os

a = 'Data/'
subjects = ["",'Nishanth']

def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    x, y, w, h = faces[0]

    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    for dir in dirs:

        label = dir.replace('Nishanth', "1"
                            )
        subject_dir_path = data_folder_path + "/" + dir
        subject_images_names = os.listdir(subject_dir_path)

    for img_name in subject_images_names:
        image_path = subject_dir_path + "/" + img_name
        image = cv2.imread(image_path)
        face, rect = detect_face(image)

        if face is not None:
            # add face to list of faces
            faces.append(face)
            # add label for this face
            labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data(a)

for i in range(0,len(labels)):
    labels[i] =  int(labels[i])

print(type(labels[0]))
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
#
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
#
#
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[labels]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

print("Predicting images...")

#load test images
test_img1 = cv2.imread("test2.jpeg")



#perform a prediction
predicted_img1 = predict(test_img1)

print("Prediction complete")

#display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()