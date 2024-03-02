import cv2
import numpy as np
import csv

# Load YOLOv3 and its weights (replace with your paths)
net = cv2.dnn.readNet(r'C:\Users\dell\Desktop\vehicle_detect\yolov3.weights', r'C:\Users\dell\Desktop\vehicle_detect\yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the region of interest (ROI) for inflow and outflow
roi_inflow_x, roi_inflow_y, roi_inflow_w, roi_inflow_h = 10, 10, 500, 500
roi_outflow_x, roi_outflow_y, roi_outflow_w, roi_outflow_h = 10, 10, 500, 500

# Map class IDs to vehicle types (replace/add as needed)
vehicle_types = {
    2: "Car",
    3: "Rickshaw",
    4: "Truck",
}

def detect_vehicles(frame, roi_x, roi_y, roi_w, roi_h):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in vehicle_types:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Count vehicles within the ROI
    inflow_count = 0
    outflow_count = 0
    vehicle_type_counts = {type_: 0 for type_ in vehicle_types.values()}
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if roi_x < x + w // 2 < roi_x + roi_w and roi_y < y + h // 2 < roi_y + roi_h:
            if x + w // 2 < roi_x + roi_w // 2:
                inflow_count += 1
            else:
                outflow_count += 1
            vehicle_type_counts[vehicle_types[class_ids[i]]] += 1

    return inflow_count, outflow_count, vehicle_type_counts

def count_vehicles_from_video(video_path, roi_x, roi_y, roi_w, roi_h):
    cap = cv2.VideoCapture(video_path)

    total_inflow_count = 0
    total_outflow_count = 0
    total_vehicle_type_counts = {type_: 0 for type_ in vehicle_types.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inflow_count, outflow_count, vehicle_type_counts = detect_vehicles(frame, roi_x, roi_y, roi_w, roi_h)
        total_inflow_count += inflow_count
        total_outflow_count += outflow_count
        for type_, count in vehicle_type_counts.items():
            total_vehicle_type_counts[type_] += count

    cap.release()
    cv2.destroyAllWindows()

    return total_inflow_count, total_outflow_count, total_vehicle_type_counts


def write_to_csv(video_path, inflow_count, outflow_count, vehicle_type_counts):
  print("oi2")
  
  # Create and open the CSV file in append mode
  with open("vehicle_counts.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    print("oi3")
    # Write header row if the file is empty
    if csvfile.tell() == 0:
      writer.writerow(["Video Path", "Inflow Count", "Outflow Count"] + list(vehicle_types.values()))

    # Write data for the current video
    writer.writerow([video_path, inflow_count, outflow_count] + [count for type_, count in vehicle_type_counts.items()])

# Example usage
inflow_video_path = r'C:\Users\dell\Desktop\vehicle_detect\inflow.mp4'
outflow_video_path = r'C:\Users\dell\Desktop\vehicle_detect\inflow.mp4'

inflow_count, outflow_count, inflow_type_counts = count_vehicles_from_video(inflow_video_path, roi_inflow_x, roi_inflow_y, roi_inflow_w, roi_inflow_h)
outflow_count, outflow_count, outflow_type_counts = count_vehicles_from_video(outflow_video_path, roi_outflow_x, roi_outflow_y, roi_outflow_w, roi_outflow_h)

efficiency = (inflow_count - outflow_count) / inflow_count * 100
# Combine and write data for both videos
#total_inflow_count = inflow_count + outflow_count
#total_outflow_count = inflow_count + outflow_count  # Assuming inflow = outflow for demonstration
#combined_type_counts = {type_: inflow_type_counts[type_] + outflow_type_counts[type_] for type_ in vehicle_types.values()}

write_to_csv("combined.mp4", efficiency)

print("Results written to 'vehicle_counts.csv'")
