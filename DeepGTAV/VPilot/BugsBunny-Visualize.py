import cv2
import json
import numpy as np
import base64
import argparse
import os
from pathlib import Path

def load_frame_index(json_path):
    """Load the frame index JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_bbox_data(bbox_json_path):
    """Load bounding box data from JSON file."""
    with open(bbox_json_path, 'r') as f:
        return json.load(f)

def load_segmentation_data(seg_json_path):
    """Load segmentation data from JSON file."""
    with open(seg_json_path, 'r') as f:
        return json.load(f)

def parse_bbox_string(bbox_str):
    """Parse bbox string into a list of bounding boxes."""
    bboxes = []
    for line in bbox_str.strip().split('\n'):
        if line:
            parts = line.split()
            if len(parts) == 5:
                class_name, x_center, y_center, width, height = parts
                bboxes.append({
                    'class_name': class_name,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height)
                })
    return bboxes

def draw_bboxes(image, bboxes):
    """Draw bounding boxes on the image."""
    h, w = image.shape[:2]
    # Color mapping for different classes
    color_map = {
        'Car': (0, 255, 0),      # Green
        'Person': (0, 0, 255),   # Red
        'Truck': (255, 0, 0),    # Blue
        # Add more classes and colors as needed
    }
    
    for bbox in bboxes:
        # Convert normalized coordinates to pixel coordinates
        x_center = int(bbox['x_center'] * w)
        y_center = int(bbox['y_center'] * h)
        width = int(bbox['width'] * w)
        height = int(bbox['height'] * h)
        
        # Calculate corner points
        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)
        
        # Get color for class (default to green if class not in color_map)
        color = color_map.get(bbox['class_name'], (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        cv2.putText(image, bbox['class_name'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def visualize_frame(base_dir, frame_index):
    """Visualize a single frame with its bounding boxes and segmentation."""
    # Construct full paths
    image_path = os.path.join(base_dir, frame_index['files']['image'])
    bbox_json_path = os.path.join(base_dir, frame_index['files']['bbox_json'])
    seg_json_path = os.path.join(base_dir, frame_index['files']['segmentation_json'])

    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Load and draw bounding boxes
    bbox_data = load_bbox_data(bbox_json_path)
    bboxes = parse_bbox_string(bbox_data['bbox2d'])
    bbox_image = draw_bboxes(original_image.copy(), bboxes)

    # Load and decode segmentation
    seg_data = load_segmentation_data(seg_json_path)
    nparr = np.frombuffer(base64.b64decode(seg_data['segmentationImage']), np.uint8)
    segmentation_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    # Create overlay
    overlay = cv2.addWeighted(bbox_image, 0.7, segmentation_image, 0.3, 0)

    # Display windows
    windows = {
        "Original with BBoxes": bbox_image,
        "Semantic Segmentation": segmentation_image,
        "Overlay": overlay
    }

    for idx, (name, img) in enumerate(windows.items()):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 600)
        cv2.moveWindow(name, idx * 820, 0)  # Position windows side by side
        cv2.imshow(name, img)

    print("Press 'q' to exit...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize GTAV frame data")
    parser.add_argument("base_dir", help="Base directory containing the dataset")
    parser.add_argument("frame_index", help="Path to frame index JSON file")
    args = parser.parse_args()

    # Load frame index
    frame_index = load_frame_index(args.frame_index)
    
    try:
        visualize_frame(args.base_dir, frame_index)
    except Exception as e:
        print(f"Error visualizing frame: {e}")

if __name__ == "__main__":
    main()