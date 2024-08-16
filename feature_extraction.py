import dlib
import cv2
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
image_path = "database/IU/18.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dets = detector(img_rgb, 1)

for detection in dets:
    
    shape = predictor(img_rgb, detection)
    
    face_descriptor = face_rec_model.compute_face_descriptor(img_rgb, shape)
    
    face_descriptor_np = np.array(face_descriptor)
    
    print("Face Descriptor:", face_descriptor_np)
    
    (x, y, w, h) = (detection.left(), detection.top(), detection.width(), detection.height())
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
