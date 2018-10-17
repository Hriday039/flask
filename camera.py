
import cv2
import sys


# In[3]:


cascPath = '/home/mrx/Downloads/Live Stream Beta 1.0/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


# In[4]:


video_capture = cv2.VideoCapture('rtsp://motasim:123456@103.234.26.174:554/user=motasim_password=123456_channel=0_stream=0.sdp')

def video():
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame
            return cv2.imencode('.jpg', frame)[1].tobytes()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()