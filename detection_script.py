import os
import cv2
import json
import time
import argparse
from datetime import datetime

from ultralytics import YOLO

def load_model(weights_path, confidence):
    model = YOLO(weights_path)
    model.conf = confidence
    model.iou = 0.5
    return model

def run_inference(frame, model, conf_thresh):
    results = model(frame, verbose=False)[0]
    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        confidence = float(box.conf[0])

        if confidence < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]

        detections.append({
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2]
        })

    return detections

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        if label == "bare_hand":
            color = (0, 0, 255)   # red
        else:
            color = (0, 255, 0)   # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Add detection count on top
    det_count = len(detections)
    cv2.putText(
        frame,
        f"Detections: {det_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),  # Yellow
        2
    )

    return frame

def save_json_log(source, filename, detections, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    log_data = {
        "source": source,
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    }

    json_path = os.path.join(
        log_dir,
        os.path.splitext(filename)[0] + ".json"
    )

    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Gloved vs Bare Hand Detection")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["image", "video", "camera"],
        help="Inference mode"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to image folder or video file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save annotated outputs"
    )

    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Directory to save JSON logs"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="model/best.pt",
        help="Path to trained YOLOv8 weights"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera ID for live mode"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display output window"
    )

    return parser.parse_args()

def process_images(input_dir, output_dir, log_dir, model, conf_thresh, display=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("No images found in input directory.")
        return

    print(f"Processing {len(image_files)} images...")

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Failed to read {img_name}, skipping.")
            continue

        detections = run_inference(frame, model, conf_thresh)
        annotated = draw_detections(frame, detections)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, annotated)

        save_json_log(
            source="image",
            filename=img_name,
            detections=detections,
            log_dir=log_dir
        )

        if display:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if display:
        cv2.destroyAllWindows()

    print(f"Completed processing {len(image_files)} images.")

def process_video(video_path, output_dir, log_dir, model, conf_thresh, display=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(output_dir, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    start_time = time.time()

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(frame, model, conf_thresh)
        annotated = draw_detections(frame, detections)

        out.write(annotated)

        save_json_log(
            source="video",
            filename=f"frame_{frame_id:06d}.jpg",
            detections=detections,
            log_dir=log_dir
        )

        if display:
            cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Video Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0

    print(f"Processed {frame_id} frames in {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    out.release()

    if display:
        cv2.destroyAllWindows()

def process_camera(camera_id, output_dir, log_dir, model, conf_thresh, display=True):
    print(f"Starting camera mode (Camera ID: {camera_id})")
    print("Press 'q' to quit\n")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    time.sleep(1)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Cannot read from camera!")
        cap.release()
        return

    print("Camera initialized successfully!\n")

    frame_id = 0
    start_time = time.time()
    
    cv2.namedWindow("Live Camera Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        detections = run_inference(frame, model, conf_thresh)
        annotated = draw_detections(frame, detections)

        if frame_id % 30 == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(frame_path, annotated)
            
            save_json_log(
                source="camera",
                filename=f"frame_{frame_id:06d}.jpg",
                detections=detections,
                log_dir=log_dir
            )

        cv2.imshow("Live Camera Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nQuitting camera mode...")
            break

        frame_id += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0

    print(f"Processed {frame_id} frames in {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    print("Loading model...")
    model = load_model(args.weights, args.confidence)
    print("Model loaded successfully!\n")

    if args.mode == "image":
        if not args.input:
            raise ValueError("--input is required for image mode")

        process_images(
            input_dir=args.input,
            output_dir=args.output,
            log_dir=args.logs,
            model=model,
            conf_thresh=args.confidence,
            display=args.display
        )

    elif args.mode == "video":
        if not args.input:
            raise ValueError("--input is required for video mode")

        process_video(
            video_path=args.input,
            output_dir=args.output,
            log_dir=args.logs,
            model=model,
            conf_thresh=args.confidence,
            display=args.display
        )

    elif args.mode == "camera":
        process_camera(
            camera_id=args.camera_id,
            output_dir=args.output,
            log_dir=args.logs,
            model=model,
            conf_thresh=args.confidence,
            display=True
        )

    else:
        raise ValueError("Invalid mode selected")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")