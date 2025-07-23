import cv2
import numpy as np
import time
import sys
import os
import torch
import torch.nn as nn
import traceback
import json
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image

if len(sys.argv) != 3:
    print("Usage: python run_clahe.py <input_video_path> <output_video_path>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

input_path = os.path.abspath(input_path)
output_path = os.path.abspath(output_path)
model_path = os.path.join(os.path.dirname(__file__), "app", "Classifier.pt")
if not os.path.exists(model_path):
    model_path = os.path.abspath("app/models/Classifier.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = convnext_base(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Classifier model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load classifier model.")
    traceback.print_exc()
    sys.exit(2)

transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor()
])

def apply_CLAHE_to_frame(frame):
    """Applies CLAHE to the Luminance channel of a BGR frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def classify_frame(frame):
    """Classifies a frame as low-light or not using the ConvNeXt model."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).item()
    return output >= 0.5  # Returns True if classified as low-light

try:
    frames_to_skip = 4 # Every 5th frame is processed - set to 0 if every frame needs to be processed
    processing_size = (640, 480) # Resized for faster processing 

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {original_width}x{original_height} @ {original_fps:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (original_width, original_height))

    frame_count = 0
    processed_count = 0
    total_processing_time = 0
    start_time = time.time()
    
    frame_timings = []
    enhanced_frames = 0

    last_processed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % (frames_to_skip + 1) == 0:
            frame_start_time = time.time()
            
            resized_frame = cv2.resize(frame, processing_size, interpolation=cv2.INTER_AREA)

            classify_start = time.time()
            is_low_light = classify_frame(resized_frame)
            classify_time = time.time() - classify_start

            enhance_start = time.time()
            if is_low_light:
                processed_frame_small = apply_CLAHE_to_frame(resized_frame)
                enhanced_frames += 1
            else:
                processed_frame_small = resized_frame
            enhance_time = time.time() - enhance_start
            
            last_processed_frame = cv2.resize(processed_frame_small, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
            
            frame_processing_time = time.time() - frame_start_time
            total_processing_time += frame_processing_time
            
            frame_timings.append({
                "frame_number": frame_count,
                "total_time": frame_processing_time,
                "classify_time": classify_time,
                "enhance_time": enhance_time,
                "is_low_light": is_low_light,
                "timestamp": time.time()
            })
            
            processed_count += 1

        if last_processed_frame is not None:
            out.write(last_processed_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")


    cap.release()
    out.release()

    elapsed_time = time.time() - start_time

    timing_output_path = output_path.replace('.mp4', '_timings.json')
    timing_data = {
        "processing_method": "clahe_with_classifier",
        "total_input_frames": frame_count,
        "frames_run_through_model": processed_count,
        "frames_enhanced_with_clahe": enhanced_frames,
        "total_processing_time_seconds": elapsed_time,
        "avg_time_per_model_run_seconds": total_processing_time / processed_count if processed_count > 0 else 0,
        "frame_by_frame_timings": frame_timings,
        "video_info": {
            "input_path": input_path,
            "output_path": output_path,
            "original_fps": original_fps,
            "output_resolution": f"{original_width}x{original_height}"
        }
    }
    
    with open(timing_output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print(f"\nProcessing complete.")
    print(f"Input frames: {frame_count} | Model runs: {processed_count}")
    print(f"Enhanced frames: {enhanced_frames}")
    print(f"Total time: {elapsed_time:.2f} sec | Avg per model run: {total_processing_time/processed_count:.4f} sec")
    print(f"Output video saved to: {output_path}")
    print(f"Timing data saved to: {timing_output_path}")

except Exception as e:
    print("\n[ERROR] An error occurred during video processing:")
    traceback.print_exc()
    sys.exit(3)
