import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/NITIN SHARMA/Desktop/sample image of , facerecognication/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_data, Labels = [], []

for i, files in enumerate(only_files):
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the images to a common size (e.g., 100x100)
    resized_image = cv2.resize(images, (100, 100))
    
    Training_data.append(np.asarray(resized_image, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))

print("Model Training Complete !")
