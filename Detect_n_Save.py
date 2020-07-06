import cv2
import os
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


dirFace = 'Nishanth'

# Create if there is no cropped face directory
if not os.path.exists(dirFace):
    os.mkdir(dirFace)
    print("Directory " , dirFace ,  " Created ")
else:
    print("Directory " , dirFace ,  " has found.")

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

count = 0
while count<=70:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    FaceFileName = "Nishanth/" + str(count) + ".jpg"
    count += 1
    cv2.imwrite(FaceFileName, img)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        ROI = gray[y:y + w, x:x + h]

        print(count)

    # Display

    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()