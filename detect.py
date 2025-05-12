import os
import cv2
import time
import csv 
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from ultralytics import YOLO

class SugarcaneNodeDetector:
    def __init__(self, model_path):
        """
        Initialize the sugarcane node detector

        Args:
            model_path: Path to trained YOLO model weights
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded model from {model_path}")
        else:
            try:
                if os.path.exists("sugarcane_model/best_model.pt"):
                    self.model = YOLO("sugarcane_model/best_model.pt")
                    print("Loaded model from sugarcane_model/best_model.pt")
                else:
                    latest_run = max(
                        [os.path.join('runs/detect', d) for d in os.listdir('runs/detect')],
                        key=os.path.getmtime
                    )
                    weights_path = os.path.join(latest_run, 'weights/best.pt')
                    self.model = YOLO(weights_path)
                    print(f"Loaded model from {weights_path}")
            except:
                print("No trained model found. Please provide a valid model path.")
                self.model = None
        
        # Initialize database connection
        self.db_conn = sqlite3.connect('sugarcane_detections.db')
        self.cursor = self.db_conn.cursor()
        
        # Create the detections table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame INTEGER,                -- Frame number where detection occurred
            timestamp TEXT,               -- Timestamp of the detection event
            node_id INTEGER,              -- ID of the node (e.g., specific location or identifier for the detection)
            confidence REAL,              -- Confidence score of the detection
            x1 REAL,                      -- Top-left corner X coordinate of the bounding box
            y1 REAL,                      -- Top-left corner Y coordinate of the bounding box
            x2 REAL,                      -- Bottom-right corner X coordinate of the bounding box
            y2 REAL,                      -- Bottom-right corner Y coordinate of the bounding box
            class_id INTEGER,             -- ID of the detected class (e.g., sugarcane node or other class)
            cut INTEGER DEFAULT 0         -- Indicator for when the bounding box is about to exit (0 by default)
        )
        ''')
        self.db_conn.commit()
    
    def __del__(self):
        """Ensure database connection is closed properly"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()

    def detect_image(self, image_path, conf_threshold=0.25, save_output=True, output_dir="results"):
        if self.model is None:
            print("No model loaded. Cannot perform detection.")
            return []

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        if save_output and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = self.model.predict(image_path, conf=conf_threshold)
        detections = []

        for idx, result in enumerate(results):
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                detections.append({
                    'id': i,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'image_path': image_path
                })

        if save_output and detections:
            result_img = results[0].plot()
            output_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, result_img)
            print(f"Detection results saved to {output_path}")

        print(f"Detected {len(detections)} nodes in {os.path.basename(image_path)}")
        for d in detections:
            print(f"  Node {d['id']}: Confidence={d['confidence']:.2f}")

        return detections

    def detect_video(self, video_path, conf_threshold=0.25, save_output=True, output_dir="results",
                     export_csv=True, show_progress=True):
        if self.model is None:
            print("No model loaded. Cannot perform detection.")
            return []

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        output_video = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(output_dir, f"output_{os.path.basename(video_path)}")
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        csv_path = None
        csv_file = None
        if export_csv:
            csv_path = os.path.join(output_dir, f"timestamps_{os.path.basename(video_path)}.csv")
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'timestamp', 'node_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'cut'])

        detections = []
        active_tracks = {}  # Track objects across frames
        frame_count = 0
        start_time = time.time()
        detected_frames = 0
        exit_margin = 20  # Pixels threshold to consider a box about to exit frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_count / fps
            hours = int(frame_time // 3600)
            minutes = int((frame_time % 3600) // 60)
            seconds = frame_time % 60
            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

            results = self.model.predict(frame, conf=conf_threshold)
            frame_detections = []
            
            # Mark currently active tracks as not seen in this frame
            for track_id in active_tracks:
                active_tracks[track_id]['seen'] = False
            
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    detected_frames += 1

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Calculate if bounding box is about to exit frame
                    about_to_exit = (
                        x1 < exit_margin or 
                        y1 < exit_margin or 
                        x2 > (width - exit_margin) or 
                        y2 > (height - exit_margin)
                    )
                    cut_value = 1 if about_to_exit else 0
                    
                    # Create unique tracking ID for this detection
                    # In a simple implementation, we'll use frame + node_id
                    track_id = f"{i}_{class_id}"
                    
                    # Check if we've seen this object before
                    if track_id in active_tracks:
                        active_tracks[track_id]['seen'] = True
                        active_tracks[track_id]['bbox'] = [x1, y1, x2, y2]
                        active_tracks[track_id]['cut'] = cut_value
                    else:
                        # New detection
                        active_tracks[track_id] = {
                            'seen': True,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'cut': cut_value,
                            'first_frame': frame_count
                        }

                    detection = {
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'node_id': i,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'cut': cut_value
                    }

                    frame_detections.append(detection)
                    
                    # Insert into SQLite database
                    if about_to_exit:
                        self.cursor.execute('''
                        INSERT INTO detections (frame, timestamp, node_id, confidence, x1, y1, x2, y2, class_id, cut)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (frame_count, timestamp, i, confidence, x1, y1, x2, y2, class_id, cut_value))
                        self.db_conn.commit()
                    
                    # Only add to CSV if about to exit or if this is a normal detection
                    if csv_file:
                        if about_to_exit:
                            csv_writer.writerow([frame_count, timestamp, i, confidence, x1, y1, x2, y2, cut_value])
                
                detections.extend(frame_detections)

            # Identify tracks that disappeared (objects that exited the frame)
            for track_id, track in list(active_tracks.items()):
                if not track['seen']:
                    # Object disappeared, remove from tracking
                    del active_tracks[track_id]

            if save_output:
                result_frame = results[0].plot() if len(frame_detections) > 0 else frame
                cv2.putText(result_frame, f"Frame: {frame_count} | Time: {timestamp}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Nodes: {len(frame_detections)} | About to Exit: {sum(1 for d in frame_detections if d['cut'] == 1)}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Highlight boxes about to exit
                for detection in frame_detections:
                    if detection['cut'] == 1:
                        bbox = detection['bbox']
                        cv2.rectangle(result_frame, 
                                     (int(bbox[0]), int(bbox[1])), 
                                     (int(bbox[2]), int(bbox[3])), 
                                     (0, 0, 255), 3)  # Red color for exit boxes
                
                output_video.write(result_frame)

            frame_count += 1
            if show_progress and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_live = frame_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_live if fps_live > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%) | " +
                      f"Speed: {fps_live:.1f} FPS | ETA: {eta:.1f} seconds")

        cap.release()
        if output_video:
            output_video.release()
        if csv_file:
            csv_file.close()

        print(f"\nDetection completed:")
        print(f"- Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")
        print(f"- Found nodes in {detected_frames} frames ({detected_frames / frame_count * 100:.1f}%)")
        print(f"- Total detections: {len(detections)}")
        if save_output:
            print(f"- Output video saved to: {output_path}")
        if csv_path:
            print(f"- CSV file saved to: {csv_path}")
        print(f"- Data saved to SQLite database: sugarcane_detections.db")

        return detections

def detect_image_batch(model_path, image_dir, conf_threshold=0.25, output_dir="results_2"):
    detector = SugarcaneNodeDetector(model_path)

    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(image_files)} images...")
    all_detections = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\nProcessing {img_file}...")
        detections = detector.detect_image(img_path, conf_threshold, output_dir=output_dir)
        all_detections.extend(detections)

    csv_path = os.path.join(output_dir, "batch_detection_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'timestamp', 'node_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        for d in all_detections:
            bbox = d['bbox']
            writer.writerow([
                os.path.basename(d['image_path']),
                d['timestamp'],
                d['id'],
                d['confidence'],
                bbox[0], bbox[1], bbox[2], bbox[3]
            ])

    print(f"\nBatch processing complete. Found {len(all_detections)} nodes in {len(image_files)} images.")
    print(f"Results saved to {csv_path}")

def create_database():
    """Create the SQLite database with the required schema"""
    # Create a connection to the SQLite database
    conn = sqlite3.connect('sugarcane_detections.db')
    cursor = conn.cursor()
    
    # SQL command to create the 'detections' table if it doesn't exist already
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame INTEGER,                -- Frame number where detection occurred
        timestamp TEXT,               -- Timestamp of the detection event
        node_id INTEGER,              -- ID of the node (e.g., specific location or identifier for the detection)
        confidence REAL,              -- Confidence score of the detection
        x1 REAL,                      -- Top-left corner X coordinate of the bounding box
        y1 REAL,                      -- Top-left corner Y coordinate of the bounding box
        x2 REAL,                      -- Bottom-right corner X coordinate of the bounding box
        y2 REAL,                      -- Bottom-right corner Y coordinate of the bounding box
        class_id INTEGER,             -- ID of the detected class (e.g., sugarcane node or other class)
        cut INTEGER DEFAULT 0         -- Indicator for when the bounding box is about to exit (0 by default)
    )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print("Database and table 'detections' created successfully")

if __name__ == "__main__":
    import argparse
    
    # Initialize database
    create_database()

    parser = argparse.ArgumentParser(description="Sugarcane Node Detection")
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video', 'batch'],
                        help='Detection mode: image, video, or batch')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (image/video) or directory (batch)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default="results",
                        help='Output directory for results')

    args = parser.parse_args()

    if args.mode == 'image':
        detector = SugarcaneNodeDetector(args.model)
        detector.detect_image(args.input, args.conf, output_dir=args.output)

    elif args.mode == 'video':
        detector = SugarcaneNodeDetector(args.model)
        detections = detector.detect_video(args.input, args.conf, output_dir=args.output)
        # You can add visualization method if needed

    elif args.mode == 'batch':
        detect_image_batch(args.model, args.input, args.conf, output_dir=args.output)

    print("\nTo run this script:")
    print("1. For video processing: python sugarcane_detector.py --mode video --input D:\sugarcane_training\test_7.mp4 --model D:\sugarcane_training\runs_1\detect\train\weights\best.pt")
    print("2. For image processing: python sugarcane_detector.py --mode image --input your_image.jpg --model path/to/model.pt")
    print("3. For batch processing: python sugarcane_detector.py --mode batch --input your_images_folder --model path/to/model.pt")