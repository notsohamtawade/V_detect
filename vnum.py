import cv2
import numpy as np
import csv

# Function to perform vehicle detection using YOLO
def detect_vehicles(image, net, layer_names, threshold=0.5):
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Run forward pass to get detections
    detections = net.forward(output_layer_names)

    # Initialize variables to count vehicles
    count_inflow = 0
    count_outflow = 0

    # Loop over the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the detected object is a vehicle (you may need to adjust class_id based on your YOLO model)
            if confidence > threshold and class_id == 2:
                # Extract bounding box coordinates
                box = obj[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype("int")

                # Calculate center coordinates of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Check if the vehicle is entering or leaving based on its position
                if center_x < width // 2:
                    count_inflow += 1
                else:
                    count_outflow += 1

                # Draw bounding box on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate efficiency percentage
    total_vehicles = count_inflow + count_outflow
    efficiency_percentage = (count_outflow / total_vehicles) * 100

    return count_inflow, count_outflow, efficiency_percentage

# User input: Number of substations
num_substations = int(input("Enter the number of substations: "))

# Load YOLO model
net = cv2.dnn.readNet(r'C:\Users\dell\Desktop\vehicle_detect\yolov3.weights', r'C:\Users\dell\Desktop\vehicle_detect\yolov3.cfg')
layer_names = net.getLayerNames()

# Open CSV file for writing
with open("vehicle_counts.csv", mode="w", newline="") as csv_file:
    fieldnames = ["Substation", "Inflow", "Outflow", "Efficiency"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header
    writer.writeheader()

    # Loop over substations
    for substation in range(1, num_substations + 1):
        entry_image_path = f"substation_{substation}_entry.jpg"
        exit_image_path = f"substation_{substation}_exit.jpg"

        # Read entry image
        entry_image = cv2.imread(entry_image_path)
        inflow, _, _ = detect_vehicles(entry_image, net, layer_names)

        # Read exit image
        exit_image = cv2.imread(exit_image_path)
        _, outflow, efficiency = detect_vehicles(exit_image, net, layer_names)

        # Write data to CSV
        writer.writerow({"Substation": f"Substation {substation}", "Inflow": inflow, "Outflow": outflow, "Efficiency": efficiency})

print("CSV file created successfully.")
