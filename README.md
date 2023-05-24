# objectdetection
To count no of objects passing through door.
import cv2

# Load the camera feed
cap = cv2.VideoCapture(0)

# Define the coordinates of the door region
door_x1, door_y1 = 200, 200
door_x2, door_y2 = 300, 400 

# Initialize the object detector
detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)

# Initialize the object count
object_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply object detection to the frame
    mask = detector.apply(frame)

    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Ignore small contours
        if area < 1500:
            continue

        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the object is passing through the door
        if x >= door_x1 and x + w <= door_x2 and y >= door_y1 and y + h <= door_y2:
            object_count += 1
            print("Object passed through the door!")

    # Draw the door region on the frame
    cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (255, 0, 0), 1)

    # Draw the object count on the frame
    cv2.putText(frame, f"Object count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows
