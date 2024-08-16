import dlib
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
image_path = "database/IU/18.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dets = detector(img_rgb, 1)

faces = dlib.full_object_detections()
for detection in dets:
    faces.append(predictor(img_rgb, detection))

aligned_faces = dlib.get_face_chips(img_rgb, faces, size=320)

for i, face in enumerate(aligned_faces):
    cv2.imshow(f"Aligned Face {i+1}", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

cv2.destroyAllWindows()
