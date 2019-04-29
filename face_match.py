import face_recognition

# Load images
img_bill_gates = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
img_unknown = face_recognition.load_image_file('./img/unknown/bill-gates-4.jpg')

# Get face encodings
bill_face_encoding = face_recognition.face_encodings(img_bill_gates)[0]
unknown_face_encoding = face_recognition.face_encodings(img_unknown)[0]

# Compare two faces encoding with tolerance of 0.5 (Lower = stricter)
results = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding, 0.5)

# Print face distance
print(f'Face Distance: {face_recognition.face_distance([bill_face_encoding], unknown_face_encoding)[0]}')

if results:
    print("This is Bill Gates")
else:
    print("This is not Bill Gates.")

