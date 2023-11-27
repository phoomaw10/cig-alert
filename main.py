import os
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


def capture_and_send_notification(frame):
    # Get the current time in Thai time zone
    now = datetime.now(pytz.utc)
    thai_timezone = pytz.timezone('Asia/Bangkok')
    now_thai = now.astimezone(thai_timezone)
    current_time = now_thai.strftime("%H:%M:%S")
    print("Current Time in Thai Time Zone =", current_time)

    # Save the captured frame as an image file
    image_path = '/Users/awatnimsiriwangso/Desktop/CigAlert/images/captured_image.jpg'
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{file_name}_{current_time}{file_extension}"
    new_image_path = os.path.join(os.path.dirname(image_path), new_file_name)

    success = cv2.imwrite(new_image_path, frame)
    if success:
            print(f"Image saved successfully at {new_image_path}")

            # Send the Line Notify message with the image
            url = 'https://notify-api.line.me/api/notify'
            token = 'vGBaIZcYZMVKTvoG9vwwPq07xDfTDFuKTaUIgZg4NU4'
            header = {'content-type': 'application/x-www-form-urlencoded',
                      'Authorization': 'Bearer '+token}

            message = {
                'message': f'Detected smoker in prohibited area at CB2301 {current_time}',
                'imageFile': open(new_image_path, 'rb')
            }

            files = {'imageFile': open('images/'+new_file_name, 'rb')}
            response = requests.post(
                url, headers=header, data=message)

            if response.status_code == 200:
                print(new_image_path)
                print(new_file_name)
                print('Message and image sent successfully.')
            else:
                print('Message and image sending failed, Error :'+str(response.status_code))
    else:
            print(f"Error saving image at {new_image_path}")


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

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('best.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ref, frame = cap.read()
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("CigAlert", frame)

        # Check if any smokers were detected
        if any(class_id == 1 for _, _, class_id, _ in detections):
            capture_and_send_notification(frame)

        if(cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
