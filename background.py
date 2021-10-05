import cv2
import numpy as np    # importing all the required libraries

video = cv2.VideoCapture('sign_sample2.mp4')    # reading sign_sample.mp4 video from the same folder


window = np.zeros([100, 700], np.uint8)   # creating a window with the following dimensions
cv2.namedWindow('window')
 
def nothing(x):
    pass
 

cv2.createTrackbar('L - h', 'window', 0, 179, nothing)
cv2.createTrackbar('U - h', 'window', 179, 179, nothing)
 
cv2.createTrackbar('L - s', 'window', 0, 255, nothing)
cv2.createTrackbar('U - s', 'window', 255, 255, nothing)
 
cv2.createTrackbar('L - v', 'window', 0, 255, nothing)
cv2.createTrackbar('U - v', 'window', 255, 255, nothing)
# creating trackbars for hsv in the window

a = 0
while(video.isOpened()):
    a = a + 1
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # reading the frame from the video and converting into hsv

    l_h = cv2.getTrackbarPos('L - h', 'window')
    u_h = cv2.getTrackbarPos('U - h', 'window')
    l_s = cv2.getTrackbarPos('L - s', 'window')
    u_s = cv2.getTrackbarPos('U - s', 'window')
    l_v = cv2.getTrackbarPos('L - v', 'window')
    u_v = cv2.getTrackbarPos('U - v', 'window')
    # reading the values of adjusted hsv values into the named variables

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    # creating a lower and upper arrays

    mask = cv2.inRange(hsv, lower_blue, upper_blue)   # masking the adjusted range of hsv values
    mask_inv= cv2.bitwise_not(mask)   # masking the remaining portion seperately

    background = cv2.bitwise_and(frame, frame, mask=mask)
    person = cv2.bitwise_and(frame, frame, mask=mask_inv)
    # assigning a window to display each masked areas
     
    cv2.imshow("background",background)
    cv2.imshow("person", person)
    # showing both the windows

    key = cv2.waitKey(100)
    # increasing the time between each frame to have some time to adjust the hsv values
    if key == ord(' '):
        break

video.release()
cv2.destroyAllWindows()   # terminating all the windows