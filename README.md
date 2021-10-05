In this project, I created a facial landmarks detector using a pre-trained model
created using IBUG300-W dataset, the dlib is used to estimate the location of 68
coordinates (x, y) that maps the facial points on a person’s face frame by frame
in a sign language video. Further I have adjusted the hsv values of the image and
seperated the background from the person, found the location of hands from the
obtained facial co-ordinates, calculated the mean value of the obtained points to get
a point on each hand of the person. This is a base to the project and this work is
able to provide quite accurate co-ordinates in a real time video.
