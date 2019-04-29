from PIL import Image
import face_recognition, os

# CONSTANTS
MODEL = 'hog'
FILE_PATH = './img/groups/team1.jpg'

# Load image file and get face locations
image = face_recognition.load_image_file(FILE_PATH)
face_locations = face_recognition.face_locations(image, 1, MODEL)

# Check if directories exists
if not os.path.isdir(f'./img/{MODEL}_results'):
    os.mkdir(f'./img/{MODEL}_results')

# Save file to result folder
for (i, face_location) in enumerate(face_locations):
    top, right, bottom, left = face_location

    # Slicing faces from the main image
    face_image = image[top:bottom, left:right]

    # Creates an image memory from an object exporting the array interface (using the buffer protocol).
    pil_image = Image.fromarray(face_image)
    # pil_image.show()
    pil_image.save(f'./img/{MODEL}_results/face_{i + 1}.jpg')
