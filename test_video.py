import numpy as np
import tensorflow as tf
import cv2
import math
# Read the graph.
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
          "boat", "traffic sign", "fire hydrant", "none", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "girrafe", "none", "backpack", "umbrella", "none", "none", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "none", "whine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "brocoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "none", "dinning table", "none", "none", "toilet", "none", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microvave", "oven", "toaster", "sink", "refrigerator", "none", "book", "vase", "teddy bear", "hair drier", "toothbrush", "none"]
graph = tf.Graph()
graph_def = tf.compat.v1.GraphDef()
detect_fn = tf.saved_model.load("extremenet")
cap = cv2.VideoCapture(0)


def distance_between_points(point1, point2, refrance_axis):
    if refrance_axis == "x":
        return math.sqrt((point1[0]-point2[0])**2)
    if refrance_axis == "y":
        return math.sqrt((point1[1]-point2[1])**2)
    if refrance_axis == "xy":
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def mid_point_between_points(point1, point2):
    mid_point_x = (point1[0]+point2[0])/2
    mid_point_y = (point1[1]+point2[1])/2
    return [mid_point_x, mid_point_y]


while True:
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    rows = frame.shape[0]
    cols = frame.shape[1]
    # out.write(frame)
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(frame_np, 0)
    detections = detect_fn(input_tensor)
    # print(out)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)
    # print(detections)
    detection_list = []
    detection_box = []
    center_data = {}
    dist_c_br_labled = {}
    for i in range(num_detections):
        class_id = int(detections["detection_classes"][i])
        score = float(detections["detection_scores"][i])
        bbox = [float(v) for v in detections["detection_boxes"][i]]
        if score > 0.5:
            if class_id not in center_data.keys():
                center_data[class_id] = []
                dist_c_br_labled[class_id] = []
                # detection_box[class_id]
            print("Class ID", class_id, "Score", score)
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            current_label_center = mid_point_between_points(
                (x, y), (right, bottom))
            dist_c_br = distance_between_points(
                current_label_center, (right, bottom), "x")
            rect_box = [int(x), int(y), int(right), int(bottom)]
            detection_list.append(class_id)
            detection_box.append(rect_box)
            # dist_c_br_labled[class_id].append(dist_c_br)
            # center_data[class_id].append(current_label_center)
    if 2 and 1 in detection_list:
        for k, ids in enumerate(detection_list):
            if ids == 1:
                cv2.rectangle(frame, (detection_box[k][0], detection_box[k][1]), (
                    detection_box[k][2], detection_box[k][3]), (255, 0, 0), thickness=2)
                cv2.putText(frame, labels[ids - 1], (detection_box[k][0], detection_box[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, 2)
            elif ids == 2:
                cv2.rectangle(frame, (detection_box[k][0], detection_box[k][1]), (
                    detection_box[k][2], detection_box[k][3]), (0, 0, 255), thickness=2)
                cv2.putText(frame, labels[ids - 1], (detection_box[k][0], detection_box[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, 2)
            else:
                cv2.rectangle(frame, (detection_box[k][0], detection_box[k][1]), (
                    detection_box[k][2], detection_box[k][3]), (0, 255, 0), thickness=2)
                cv2.putText(frame, labels[ids - 1], (detection_box[k][0], detection_box[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, 2)
    cv2.imshow('TensorFlow MobileNet-SSD', frame)
    if cv2.waitKey(90) & 0xFF == ord("q"):
        break
