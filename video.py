import cv2
firstFrame = None
# Start Video
video = cv2.VideoCapture(0)
while True:
    # Frame details obtained from the video
    check, frame = video.read()

    # Grayscale Image of frame
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grayImage = cv2.GaussianBlur(grayImage,(21,21),0)

    # First Frame is stored for comparison

    if firstFrame is None:
        firstFrame = grayImage
        continue

    # Difference between current frame and first frame
    frameDiff = cv2.absdiff(firstFrame, grayImage)

    # Threshold Image
    thresholdFrame = cv2.threshold(frameDiff,30,255,cv2.THRESH_BINARY)[1]

    # Threshold Image is dilated to find objects within frame
    thresholdFrame = cv2.dilate(thresholdFrame,None,iterations=2)

    # contour (objects) are detected in current frame
    (cntrs,_) = cv2.findContours(thresholdFrame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    for contours in cntrs:

        # Object is considered only if contour area < 1000px
        if cv2.contourArea(contours) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(contours)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),3)

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    #Press "Q" on your keyboard to stop the camera
    if key == ord('q'):
        break

# Video End
video.release()
cv2.destroyAllWindows()