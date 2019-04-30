import face_recognition

img = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
face_landmarks = face_recognition.face_landmarks(img)
print(face_landmarks)