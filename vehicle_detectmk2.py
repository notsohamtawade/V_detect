import cv2
import numpy as np
import csv
import time

# Load YOLOv3 and its weights (replace with your paths)
net = cv2.dnn.readNet(r'C:\Users\dell\Desktop\vehicle_detect\yolov3.weights', r'C:\Users\dell\Desktop\vehicle_detect\yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("1")

# Define the region of interest (ROI) for inflow and outflow
roi_inflow_x, roi_inflow_y, roi_inflow_w, roi_inflow_h = 152, 152, 304, 304
roi_outflow_x, roi_outflow_y, roi_outflow_w, roi_outflow_h = 152, 152, 304, 304
print("2")
# Map class IDs to vehicle types (replace/add as needed)
vehicle_types = {
    2: "Car",
    
    
}
print("3")
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
    print("5")

def count_vehicles_from_video(video_path, roi_x, roi_y, roi_w, roi_h):
    print("video_path",video_path)
    print(roi_x,roi_y,roi_w,roi_h)
    cap = cv2.VideoCapture(video_path)
    print("6")
    total_inflow_count = 0
    total_outflow_count = 0
    total_vehicle_type_counts = {type_: 0 for type_ in vehicle_types.values()}
    print("7.2")
    while True:
        ret, frame = cap.read()
        print("ret:",ret)
        if not ret:
         break

        inflow_count, outflow_count, vehicle_type_counts = detect_vehicles(frame, roi_x, roi_y, roi_w, roi_h)
        print("7.3")
        total_inflow_count += inflow_count
        total_outflow_count += outflow_count
        print("7.4")
        for type_, count in vehicle_type_counts.items():
         print("Type:", type_, "Count:", count)   
         total_vehicle_type_counts[type_] += count
        
         print("7.5")
              
    print("7.1")
    cap.release()
    cv2.destroyAllWindows()
    print(total_inflow_count,total_outflow_count,total_vehicle_type_counts)
    return total_inflow_count, total_outflow_count, total_vehicle_type_counts
    print("7")


def write_to_csv(video_path, inflow_count, outflow_count, vehicle_type_counts):
    # Create and open the CSV file in append mode
    print("8")
    with open("vehicle_counts.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row if the file is empty
        if csvfile.tell() == 0:
            writer.writerow(["Video Path", "Inflow Count", "Outflow Count"] + list(vehicle_types.values()))

        # Write data for the current video
        writer.writerow([video_path, inflow_count, outflow_count] + [count for type_, count in vehicle_type_counts.items()])
    print("9")

# Define time interval for data collection (5 minutes = 300 seconds)
collection_interval_seconds = 300

# Keep track of the start time for data collection
start_time = time.time()

# Initialize variables to track data within the time interval
interval_inflow_count = 0
interval_outflow_count = 0
interval_vehicle_type_counts = {type_: 0 for type_ in vehicle_types.values()}
print("10")
# Main loop for video processing
inflow_video_path = r'C:\Users\dell\Desktop\vehicle_detect\inflow.mp4'
outflow_video_path = r'C:\Users\dell\Desktop\vehicle_detect\outflow.mp4'
while True:
    # Check if it's time to write data to CSV
    if time.time() - start_time >= collection_interval_seconds:
        # Write data to CSV
        write_to_csv("combined.mp4", interval_inflow_count, interval_outflow_count, interval_vehicle_type_counts)
        
        # Reset variables for the next time interval
        interval_inflow_count = 0
        interval_outflow_count = 0
        interval_vehicle_type_counts = {type_: 0 for type_ in vehicle_types.values()}
        
        # Update start time for the next time interval
        start_time = time.time()
    print("11")
    # Process inflow video
    inflow_count, outflow_count, inflow_type_counts = count_vehicles_from_video(inflow_video_path, roi_inflow_x, roi_inflow_y, roi_inflow_w, roi_inflow_h)
    interval_inflow_count += inflow_count
    interval_outflow_count += outflow_count
    for type_, count in inflow_type_counts.items():
        interval_vehicle_type_counts[type_] += count
    print("12")
    # Process outflow video
    outflow_count, outflow_count, outflow_type_counts = count_vehicles_from_video(outflow_video_path, roi_outflow_x, roi_outflow_y, roi_outflow_w, roi_outflow_h)
    interval_inflow_count += inflow_count
    interval_outflow_count += outflow_count
    for type_, count in outflow_type_counts.items():
        interval_vehicle_type_counts[type_] += count
    print("13")
# Write remaining data to CSV at the end of video processing
write_to_csv("combined.mp4", interval_inflow_count, interval_outflow_count, interval_vehicle_type_counts)
print("done!")
