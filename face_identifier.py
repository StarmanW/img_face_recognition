import face_recognition, os
from PIL import Image, ImageDraw

# CONSTANTS
KNOWN_IMAGE_DIRECTORY = './img/known'
TEST_IMAGE_DIRECTORY = './img/groups/bill-steve-elon.jpg'
OUTPUT_DIRECTORY = f'./img/face_identifier_results'
RESULT_FILENAME = f'{os.path.basename(__file__)}_result.jpg'
SUPPORTED_IMG_FORMATS = [
    'JPG',
    'JPEG',
    'PNG',
    'BMP',
]
face_encodings = {}


# Initialize face encodings from known faces
def init_face_encodings(img_path):
    # Encode directory
    directory = os.fsencode(img_path)

    # Loop through the list of files in the given directory
    for file in os.listdir(directory):
        # Decode file and split based on '.' delimeter
        filename = os.fsdecode(file).split('.')

        # Check for supported image file
        if SUPPORTED_IMG_FORMATS.index(filename[1].upper()) != -1:
            # Get face encoding
            face_encoding = \
                face_recognition.face_encodings(face_recognition.load_image_file(f'{img_path}/{os.fsdecode(file)}'))[0]

            # Add face encoding to dictionary using image name as key
            if (filename[0] not in face_encodings.keys()):
                face_encodings[filename[0]] = face_encoding


# Save image to path
def save_image(path, filename):
    # Check if directories exists
    if not os.path.isdir(path):
        os.mkdir(path)
    pil_image.save(f'{path}/{filename}')


init_face_encodings(KNOWN_IMAGE_DIRECTORY)

# Load test image and get face encodings
test_image = face_recognition.load_image_file(TEST_IMAGE_DIRECTORY)
face_locations = face_recognition.face_locations(test_image, 1, 'cnn')
test_img_face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert image to PIL Image
pil_image = Image.fromarray(test_image)

# Create drawable image
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for (top, right, bottom, left), face_encoding in zip(face_locations, test_img_face_encodings):
    matches = face_recognition.compare_faces(list(face_encodings.values()), face_encoding)
    name = "Unknown Person"

    # If match
    if True in matches:
        index = matches.index(True)
        name = list(face_encodings.keys())[index]

    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))

    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 255, 0), outline=(255, 255, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))

# Release draw memory resources
del draw

# Display image
pil_image.show()

# Save image
save_image(OUTPUT_DIRECTORY, RESULT_FILENAME)
