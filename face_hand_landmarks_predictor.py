import cv2
import dlib
import numpy as np    # importing all the required libraries

video = cv2.VideoCapture('sign_sample1.mp4')    # reading sign_sample.mp4 video from the same folder 
detector = dlib.get_frontal_face_detector()     # run the face detector from dlib
predictor = dlib.shape_predictor("C:\\Users\\dakhi\\Documents\\Python_Scripts\\samples\\face_detection_landmarks\\shape_predictor_68_face_landmarks.dat")   
# importing a pre trained landmarks model into the predictor

_, person = video.read()    # reading each frame of the video
gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)    # converting the frame into gray scale
faces = detector(gray)    # gray scaled images are now processed through the detector
i = 0
for face in faces:    # loop to make sure all the frames of the video are detected
    i = i + 1
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    # finding the boundaries of the face to eliminate it during the hand detection using skin color segmentation    

    landmarks = predictor(gray, face)   # landmarks of the face are obtained using the predictor and the pre trained model imported into it
    
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(person, (x, y), 1, (0, 255, 0), -1)
    # drawing a small circle(simply a dot) at each of the 68 landmarks on the face 
 
person=cv2.flip(person,1)
kernel = np.ones((3,3),np.uint8)
roi=person[0:500, 0:500]    # creating a roi to have a fixed working area for every video 
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)    # converting frame into hsv 
lower_skin = np.array([0,48,80], dtype=np.uint8)
upper_skin = np.array([20,255,255], dtype=np.uint8)   # a region of hsv values for skin color is found out using tuning method from background.py
mask = cv2.inRange(hsv, lower_skin, upper_skin)   # extracting skin color image
hands = np.argwhere(mask == [255])    # making every skin colored area into white
handR_x = []
handR_y = []
handL_x = []
handL_y = []
# creating an empty array for each coordinate of both the hands
for h in hands :
    if 0 < h[0] < (x1+100):
        handR_x.append(h[0])
    if (x2-150) < h[0] < 500:
        handL_x.append(h[0])
    if h[1] > (y2+150):
        handR_y.append(h[1])
        handL_y.append(h[1])
# assigning the coordinates in the sides of the hand excepting the facial area to their respective arrays

h_r_x = sum(handR_x)//len(handR_x)
h_r_y = sum(handR_y)//len(handR_y)
h_l_x = sum(handL_x)//len(handL_x)
h_l_y = sum(handL_y)//len(handL_y)
# finding average to have a single average point on each hand

cv2.circle(person, (h_r_x, h_r_y), 1, (0, 255, 0), -1)
cv2.circle(person, (h_l_x, h_l_y), 1, (0, 255, 0), -1)
cv2.imshow("person", person)    # marking the hand coordinates also and showing the final image with all the coordinates
#cv2.imwrite("C:\\Users\\dakhi\\Documents\\Python_Scripts\\samples\\result.jpg", person)
key = cv2.waitKey()
if key == ord(' '):
    exit
# the final image with all the 70 corodinates is displayed till the space is pressed

print("Detected", len(landmarks.parts()), "landmarks in face and 2 hand landmarks.")
tup_out = (('jaw1', landmarks.part(0).x, landmarks.part(0).y), ('jaw2', landmarks.part(1).x, landmarks.part(1).y), ('jaw3', landmarks.part(2).x, landmarks.part(2).y), ('jaw4', landmarks.part(3).x, landmarks.part(3).y), \
        ('jaw5', landmarks.part(4).x, landmarks.part(4).y)), ('jaw6', landmarks.part(5).x, landmarks.part(5).y), ('jaw7', landmarks.part(6).x, landmarks.part(6).y), ('jaw8', landmarks.part(7).x, landmarks.part(7).y), \
        ('jaw9', landmarks.part(8).x, landmarks.part(8).y), ('jaw10', landmarks.part(9).x, landmarks.part(9).y), ('jaw11', landmarks.part(10).x, landmarks.part(10).y), ('jaw12', landmarks.part(11).x, landmarks.part(11).y), \
        ('jaw13', landmarks.part(12).x, landmarks.part(12).y), ('jaw14', landmarks.part(13).x, landmarks.part(13).y), ('jaw15', landmarks.part(14).x, landmarks.part(14).y), ('jaw16', landmarks.part(15).x, landmarks.part(15).y), \
        ('jaw17', landmarks.part(16).x, landmarks.part(16).y), ('browR18', landmarks.part(17).x, landmarks.part(17).y), ('browR19', landmarks.part(18).x, landmarks.part(18).y), ('browR20', landmarks.part(19).x, landmarks.part(19).y), \
        ('browR21', landmarks.part(20).x, landmarks.part(20).y), ('browR22', landmarks.part(21).x, landmarks.part(21).y), ('browL23', landmarks.part(22).x, landmarks.part(22).y), ('browL24', landmarks.part(23).x, landmarks.part(23).y), \
        ('browL25', landmarks.part(24).x, landmarks.part(24).y), ('browL26', landmarks.part(25).x, landmarks.part(25).y), ('browL27', landmarks.part(26).x, landmarks.part(26).y), ('nose28', landmarks.part(27).x, landmarks.part(27).y),  \
        ('nose29', landmarks.part(28).x, landmarks.part(28).y), ('nose30', landmarks.part(29).x, landmarks.part(29).y), ('nose31', landmarks.part(30).x, landmarks.part(30).y), ('nose32', landmarks.part(31).x, landmarks.part(31).y), \
        ('nose33', landmarks.part(32).x, landmarks.part(32).y), ('nose34', landmarks.part(33).x, landmarks.part(33).y), ('nose35', landmarks.part(34).x, landmarks.part(34).y), ('nose36', landmarks.part(35).x, landmarks.part(35).y), \
        ('eyeR37', landmarks.part(36).x, landmarks.part(36).y), ('eyeR38', landmarks.part(37).x, landmarks.part(37).y), ('eyeR39', landmarks.part(38).x, landmarks.part(38).y), ('eyeR40', landmarks.part(39).x, landmarks.part(39).y), \
        ('eyeR41', landmarks.part(40).x, landmarks.part(40).y), ('eyeR42', landmarks.part(41).x, landmarks.part(41).y), ('eyeL43', landmarks.part(42).x, landmarks.part(42).y), ('eyeL44', landmarks.part(43).x, landmarks.part(43).y), \
        ('eyeL45', landmarks.part(44).x, landmarks.part(44).y), ('eyeL46', landmarks.part(45).x, landmarks.part(45).y), ('eyeL47', landmarks.part(46).x, landmarks.part(46).y), ('eyeL48', landmarks.part(47).x, landmarks.part(47).y), \
        ('lipO49', landmarks.part(48).x, landmarks.part(48).y), ('lipO50', landmarks.part(49).x, landmarks.part(49).y), ('lipO51', landmarks.part(50).x, landmarks.part(50).y), ('lipO52', landmarks.part(51).x, landmarks.part(51).y), \
        ('lipO53', landmarks.part(52).x, landmarks.part(52).y), ('lipO54', landmarks.part(53).x, landmarks.part(53).y), ('lipO55', landmarks.part(54).x, landmarks.part(54).y), ('lipO56', landmarks.part(55).x, landmarks.part(55).y), \
        ('lipO57', landmarks.part(56).x, landmarks.part(56).y), ('lipO58', landmarks.part(57).x, landmarks.part(57).y), ('lipO59', landmarks.part(58).x, landmarks.part(58).y), ('lipO60', landmarks.part(59).x, landmarks.part(59).y), \
        ('lipI61', landmarks.part(60).x, landmarks.part(60).y), ('lipI62', landmarks.part(61).x, landmarks.part(61).y), ('lipI63', landmarks.part(62).x, landmarks.part(62).y), ('lipI64', landmarks.part(63).x, landmarks.part(63).y), \
        ('lipI65', landmarks.part(64).x, landmarks.part(64).y), ('lipI66', landmarks.part(65).x, landmarks.part(65).y), ('lipI67', landmarks.part(66).x, landmarks.part(66).y), ('lipI68', landmarks.part(67).x, landmarks.part(67).y), \
        ('handR', sum(handR_x)/len(handR_x), sum(handR_y)/len(handR_y)), ('handL', sum(handL_x)/len(handL_x), sum(handL_y)/len(handL_y))
        
print("Landmark co-ordinates are: ", tup_out[0:68])
print("Total number of faces detected: ", i)        
# printing the result

video.release()
cv2.destroyAllWindows()   # terminating all the windows