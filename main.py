import os
import subprocess
import time
import cv2
import argparse
import requests
import datetime

import supervision as sv
from ultralytics import YOLO

from pydub import AudioSegment
from pydub.playback import play

from datetime import datetime
import pytz


cooldown_start_time = None
cooldown_duration = 3


def capture_and_send_notification(frame):
    global cooldown_start_time

    if cooldown_start_time is not None and (time.time() - cooldown_start_time) < cooldown_duration:
        print("Waiting for cooldown period...")
        return

    # Get the current time in Thai time zone
    now = datetime.now(pytz.utc)
    thai_timezone = pytz.timezone('Asia/Bangkok')
    now_thai = now.astimezone(thai_timezone)
    current_time = now_thai.strftime("%H:%M:%S")
    print("Current Time in Thai Time Zone =", current_time)

    # Save the captured frame as an image file
    image_path = '/Users/banu/Documents/GitHub/cig-alert2/images/captured_image.jpg'
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{file_name}_{current_time}{file_extension}"
    new_image_path = os.path.join(os.path.dirname(image_path), new_file_name)

    success = cv2.imwrite(new_image_path, frame)
    if success:
        print(f"Image saved successfully at {new_image_path}")

        # Send the Line Notify message with the image
        url = 'https://notify-api.line.me/api/notify'
        # Old svvhUZwqxL9o3XVysml5qDZA9tcR3NKmVcAtfnvkEQA
        token = '527ur48rgpF5IyNprCKJlVfIKGz0MdANv3deUc4WLPg'
        header = {'content-type': 'application/x-www-form-urlencoded',
                  'Authorization': 'Bearer '+token}

        message = {
            'message': f'Detected smoker in prohibited area at CB2301 {current_time}',
            # 'imageFile': open(new_image_path, 'rb')
        }

        if os.path.exists(new_image_path):
            curl_command = [
                'curl',
                '-X', 'POST',
                '-H', f'Authorization: Bearer {token}',
                '-F', 'message=Detected smoker in prohibited area at CB2301 ' + current_time,
                '-F', f'imageFile=@{new_image_path}',
                'https://notify-api.line.me/api/notify'
            ]
        try:
            subprocess.run(curl_command, check=True)
            print(new_image_path)
            print(new_file_name)
            print('Message and image sent successfully.')
            cooldown_start_time = time.time()
        except subprocess.CalledProcessError as e:
            print(new_image_path)
            print('Message and image sending failed, Error :' + str(e))
    else:
        print(f"Image file '{new_image_path}' does not exist.")
    # cooldown_start_time = time.time()


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('best.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    face_detected = False
    cigarette_detected = False
    detection_start_time = None

    while True:
        ref, frame = cap.read()

        # equalized frame
        b, g, r = cv2.split(frame)
        eq_b = cv2.equalizeHist(b)
        eq_g = cv2.equalizeHist(g)
        eq_r = cv2.equalizeHist(r)

        equalized_frame = cv2.merge((eq_b, eq_g, eq_r))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_stacked = cv2.merge([gray_frame] * 3)
        result = model(gray_frame_stacked)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("CigAlert", frame)

        # Check if any smokers were detected
        # if any(class_id == 1 for _, _, _, class_id, _ in detections):
        # capture_and_send_notification(frame)

        if any(confidence > 0.7 and class_id == 1 for _, _, confidence, class_id, _ in detections):
            face_detected = True

        if any(confidence > 0.7 and class_id == 0 for _, _, confidence, class_id, _ in detections):
            cigarette_detected = True

        def display_notification(frame, text):
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if face_detected and cigarette_detected:
            if detection_start_time is None:
                detection_start_time = time.time()
                display_notification(frame, "Smoking Detected!")
            elif time.time() - detection_start_time >= 2:
                capture_and_send_notification(frame)
                face_detected = False
                cigarette_detected = False
                detection_start_time = None
                display_notification(frame, "Smoking Detected!")
        else:
            detection_start_time = None
            face_detected = False
            cigarette_detected = False
        cv2.imshow("CigAlert", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
