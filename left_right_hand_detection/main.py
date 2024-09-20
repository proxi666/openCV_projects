# Import libraries
import cv2
import mediapipe as mp

# Used to convert protobuf message  
# to a dictionary. 
from google.protobuf.json_format import MessageToDict


# Initilizing the model 
mpHands = mp.solutions.hands
hands = mpHands.Hands (
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence = 0.75,
    min_tracking_confidence = 0.75,
    max_num_hands = 2
)

cap = cv2.VideoCapture(0)

while True:

    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        # If both hands are present in the image
        if len(results.multi_handedness) == 2:
            # Disply 'both hands TEXT' on the image
            cv2.putText(img, 'Both Hands', (250, 250),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (0, 255, 0), 2)
            
        # If any one hand is present
        else:
            for i in results.multi_handedness:

                # Return weather it is right or left hand
                label = MessageToDict(i)[
                    'classification'][0]['label']
                
                if label == 'Left':

                    # Disply "left hand TEXT" on the left side of the window
                    cv2.putText(img, label+' Hand', (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)
                    
                if label == 'Right':

                    #Disply 'right hand' on the right side of the window
                    cv2.putText(img, label+' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                    
# DESTROY WINDOW
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break