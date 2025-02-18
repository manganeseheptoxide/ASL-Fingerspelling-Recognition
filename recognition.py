import mediapipe as mp
import cv2
from data_processing import *
from data_collection import *
from trained_models import *

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
keras_model = KerasModel(language='fsl')
xgb_model = XGBoostModel(language='fsl')

def detect():

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while True:
    
            ret, frame = cap.read()
            if not ret:
                _status = False
                break
            image, _landmarks_list, _landmark_connections = detect_upperbody(frame, hands)
            if _landmarks_list and _landmarks_list.landmark:
                mp_drawing.draw_landmarks(image, _landmarks_list, _landmark_connections)
                
                # Calculate the bounding box
                h, w, _ = image.shape
                buffer = 0.1
                xs = [landmark.x for landmark in _landmarks_list.landmark]
                ys = [landmark.y for landmark in _landmarks_list.landmark]
                
                x_min = min(xs) * w * (1 - buffer)
                y_min = min(ys) * h * (1 - buffer)
                x_max = max(xs) * w * (1 + buffer)
                y_max = max(ys) * h * (1 + buffer)
                # data = 
                # Draw the bounding box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                landmark_list = center_xyzlandmarks(_landmarks_list)
                data = []
                for landmark in landmark_list.landmark:
                    data += [landmark.x, landmark.y, landmark.z]
                # print(data)
                label = keras_model.detect(data)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = int(x_min)
                text_y = int(y_min)
                cv2.rectangle(image, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y + 10), (0, 255, 0), -1)
                cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

detect()



