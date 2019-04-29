# Import
import face_recognition

MODEL = 'cnn'

# Groups image file name
images = [
    'bill-steve.jpg',
    'bill-steve-elon.jpg',
    'team1.jpg',
    'team2.jpg'
]

# Load image
image = face_recognition.load_image_file(f'./img/groups/{images[3]}')

# Get face locations array
# 1 for default upsample times
# CNN for more accuracy but consume more CPUs
face_locations = face_recognition.face_locations(image, 1, MODEL)

# Get the array of coordinates for each face
print(f'There are a total of {len(face_locations)} people in the image.')

# For loop print face locations of the image
for (i, face_location) in enumerate(face_locations):
    print(f'Face no.{i + 1}\n'
          f'----------\n'
          f'TOP: {face_location[0]}\n'
          f'RIGHT: {face_location[1]}\n'
          f'BOTTOM: {face_location[2]}\n'
          f'LEFT: {face_location[3]}\n')
