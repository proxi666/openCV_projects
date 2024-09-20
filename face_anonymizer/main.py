import cv2
import mediapipe as mp
import os
import argparse


def process_img(img, face_detection):
    H, W, _ = img.shape  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = face_detection.process(img_rgb)

    if out.detections:  # Changed from 'is not None'
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box 

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # BLUR FACE
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (10,10))

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default='video')
args.add_argument("--filepath", default='input_images/video.mp4')  # Corrected typo

args = args.parse_args()
out_path = 'output_images'
if not os.path.exists(out_path):
    os.makedirs(out_path)

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode == "image":
        img = cv2.imread(args.filepath)
        if img is None:
            print(f"Error: Unable to read image file: {args.filepath}")
        else:
            img = process_img(img, face_detection)
            cv2.imwrite(os.path.join(out_path, 'blurred_img.png'), img)

    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.filepath)
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {args.filepath}")
        else:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read the first frame from: {args.filepath}")
            else:
                output_video = cv2.VideoWriter(os.path.join(out_path, 'output.mp4'),
                                               cv2.VideoWriter_fourcc(*'MP4V'),
                                               25,
                                               (frame.shape[1], frame.shape[0]))
                
                while ret:
                    frame = process_img(frame, face_detection)
                    output_video.write(frame)
                    ret, frame = cap.read()

                output_video.release()
            cap.release()

    else:
        print(f"Error: Invalid mode '{args.mode}'. Use 'image' or 'video'.")

cv2.destroyAllWindows()