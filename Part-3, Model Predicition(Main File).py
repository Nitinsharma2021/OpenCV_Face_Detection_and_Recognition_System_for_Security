import cv2
import os

import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/NITIN SHARMA/Desktop/sample image of , facerecognication/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_data, Labels = [], []
label_id = 0
label_map = {}  # Map names to label IDs

width, height = 200, 200  # Desired dimensions for resized images

for i, file_name in enumerate(only_files):
    image_path = data_path + only_files[i]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (width, height))
    Training_data.append(image_resized)
    
    label = file_name.split('.')[0]  # Assuming file names are like '1.jpg', '2.jpg', etc.
    if label not in label_map:
        label_map[label] = label_id
        label_id += 1
    Labels.append(label_map[label])

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))

print("Model Training Complete !")

face_classifier = cv2.CascadeClassifier('C:\\Users\\NITIN SHARMA\\Desktop\\New folder (2)\\Face-Unlock-and-Lock-Recogination-System-main\\haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

cap = cv2.VideoCapture(0)

locked_session_count = 1
captured_images_count = 0
max_captured_images = 10  # Maximum number of images to capture in a locked session

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        if face is not None:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% Confidence it is User'

                if confidence > 80:
                    cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    captured_images_count = 0  # Reset the captured images count when unlocked
                else:
                    if captured_images_count < max_captured_images:
                        cv2.putText(image, f"Locked - Capturing ({captured_images_count+1}/{max_captured_images})", 
                                    (150, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        
                        # Save an image of the person when locked
                        locked_face_path = 'C:/Users/NITIN SHARMA/Desktop/sample image of , facerecognication/locked_image/'
                        locked_session_path = os.path.join(locked_face_path, f'locked_session_{locked_session_count}/')
                        os.makedirs(locked_session_path, exist_ok=True)
                        locked_face_filename = os.path.join(locked_session_path, f'locked_image_{captured_images_count + 1}.jpg')
                        cv2.imwrite(locked_face_filename, frame)
                        captured_images_count += 1
                        print("Locked! Image saved as", locked_face_filename)
                    else:
                        cv2.putText(image, "Locked - Images Captured", (150, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Face Not Recognized", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "No Face Detected", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Face Cropper", image)

    except Exception as e:
        print(str(e))
        cv2.putText(image, "Error", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Cropper", image)
        pass

    if cv2.waitKey(1) == 13:
        break

    if cv2.waitKey(1) == ord('q'):
        locked_session_count += 1

cap.release()
cv2.destroyAllWindows()
