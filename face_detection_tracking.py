import argparse
import cv2
import random
import time

def main():
    i = 0
    parser = argparse.ArgumentParser(description='Detect face in video or webcam')
    parser.add_argument('--video', help='Video source')
    parser.add_argument('--out', help='Output video path')
    parser.add_argument('--skipframe', nargs='?', const=True, 
                        help='Skip frame when do not keep up frames of video')
    parser.add_argument('--waitframe', nargs='?', const=True, 
                        help='Wait frame when process faster frames of video')
    args = parser.parse_args()
    is_skipframe = args.skipframe
    is_waitframe = args.waitframe
    output_path = args.out
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
    face_tracker = {}
    frame_counter = 0
    face_count = 0
    skip_frame = 0
    base_tracker = cv2.TrackerKCF_create
    # base_tracker = cv2.TrackerMOSSE_create
    output_size = (775, 600)
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path,fourcc, 1/frame_duration, output_size)
    while capture.isOpened():
        #Retrieve the latest image from the webcam
        start_frame_time = time.time()
        rc,full_size_base_image = capture.read()
        if skip_frame > 0:
            skip_frame -= 1
            continue
        frame_counter += 1
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

        tracker_to_del = []
        bboxes = []
        for fid in face_tracker.keys():
            ok, bbox = face_tracker[fid].update(base_image)
            if ok:
                bboxes.append((fid, bbox))
            else:
                tracker_to_del.append(fid)
        for fid in tracker_to_del:
            del face_tracker[fid]
        
        # scale factor = 1.3: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors = 5 Parameter specifying how many neighbors each candidate rectangle
        # should have to retain it.
        if frame_counter % 3 == 0:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                x_center = x + 0.5*w
                y_center = y + 0.5*h
                fid_match = None
                for fid, bbox in bboxes:
                    x_tracked, y_tracked, w_tracked, h_tracked = bbox
                    x_center_tracked = x_tracked + 0.5*w_tracked
                    y_center_tracked = y_tracked + 0.5*h_tracked
                    if x_tracked < x_center < x_tracked + w_tracked \
                            and y_tracked < y_center < y_tracked + h_tracked \
                            and x < x_center_tracked < x + w \
                            and y < y_center_tracked < y + h:
                        fid_match = fid
                        break
                if fid_match is None:
                    new_tracker = base_tracker()
                    new_tracker.init(base_image, (x, y, w, h))
                    face_count += 1
                    face_tracker[face_count] = new_tracker
                    bboxes.append((face_count, (x, y, w, h)))
                else:
                    new_tracker = base_tracker()
                    new_tracker.init(base_image, (x, y, w, h))
                    face_tracker[fid_match] = new_tracker
        for fid, (x, y, w, h) in bboxes:
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)), rectangle_color, 2)
        if pressed_key == ord('f'):
            i += 1
            cv2.imwrite('./fail_detect/img_{}.jpg'.format(i), full_size_base_image)
            cv2.imwrite('./fail_detect/img_{}_fail.jpg'.format(i), result_image)
            print('fail saved')
        large_result = cv2.resize(result_image, output_size)

        #Finally, we want to show the images on the screen
        # cv2.imshow("base-image", base_image)
        cv2.imshow("video", large_result)
        end_frame_time = time.time()
        process_time = end_frame_time - start_frame_time
        if process_time < frame_duration:
            if is_waitframe:
                time.sleep(frame_duration - process_time)
        elif is_skipframe:
            is_skipframe = int(process_time/frame_duration) - 1
        if output_path is not None:
            out.write(large_result)
if __name__ == '__main__':
    main()