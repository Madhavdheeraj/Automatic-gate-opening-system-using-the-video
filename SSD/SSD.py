import cv2
import numpy as np

# Load the pre-trained SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Open the video capture
cap = cv2.VideoCapture("highway.mp4")

# Check if the video capture is opened correctly
if not cap.isOpened():
    print("Error opening video stream or file")

class_names = ["BACKGROUND", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust this value to control the confidence threshold
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label
            label = "{}: {:.2f}%".format(class_names[class_id], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Display the frame with detections
    cv2.imshow("Frame", frame)

    # Wait for 'q' key to quit
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Increase the delay for slow motion
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
