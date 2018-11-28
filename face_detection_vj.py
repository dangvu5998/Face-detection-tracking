import argparse
import cv2
import random
import time

def main():
    i = 0
    parser = argparse.ArgumentParser(description='Detect face in video or webcam')
    parser.add_argument('--video', help='Video source')
    parser.add_argument('--out', help='Output video path')
    args = parser.parse_args()
    if args.video is not None:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    height_orig = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width_orig = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_duration = 1/capture.get(cv2.CAP_PROP_FPS)
    #The deisred output width and height
    width = 600
    scale = width/width_orig
    rectangle_color = (0,165,255)
    output_size = (775, 600)
    output_path = args.out
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path,fourcc, 1/frame_duration, output_size)
    while capture.isOpened():
        #Retrieve the latest image from the webcam
        rc,full_size_base_image = capture.read()
        #Resize the image to 640x480
        base_image = cv2.resize(full_size_base_image, None, fx=scale, fy=scale)
        pressed_key = cv2.waitKey(2)
        if pressed_key == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            exit(0)
        result_image = base_image.copy()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(full_size_base_image, cv2.COLOR_BGR2GRAY)
        # scale factor = 1.3: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors = 5 Parameter specifying how many neighbors each candidate rectangle
        # should have to retain it.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), rectangle_color, 2)
        large_result = cv2.resize(result_image, output_size)

        #Finally, we want to show the images on the screen
        # cv2.imshow("base-image", base_image)
        cv2.imshow("video", large_result)
        if output_path is not None:
            out.write(large_result)

if __name__ == '__main__':
    main()