import os
import argparse
import glob
from tqdm import tqdm
from nudenet import NudeDetector
import json

def get_leaf_directories(root_dir):
    """
    Recursively finds all directories that contain no subdirectories (leaf nodes).
    """
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:  # No subdirectories, so it's a leaf directory
            leaf_dirs.append(dirpath)
    return leaf_dirs

def detect_images_in_folder(detector, folder_path, extensions, threshold):
    """
    Detects unsafe content in images within a specific folder.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
    if not image_paths:
        return {}

    results = {}
    # NudeDetector can process a list of paths, but processing one by one allows better error handling per file
    # or batch processing if needed. Here we process one by one for simplicity and progress tracking.
    for img_path in image_paths:
        try:
            detection_result = detector.detect(img_path)
            # Filter results based on threshold if needed, or just store all
            filtered_result = [
                res for res in detection_result 
                if res.get('score', 0) >= threshold
            ]
            
            if filtered_result:
                results[os.path.basename(img_path)] = filtered_result
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Run NudeNet detection on images in a directory structure.")
    parser.add_argument("--input_dir", type=str, default='E:\phd//4\code\VINE\W_bench_en\PGD_base_attack', help="Root directory containing images.")
    parser.add_argument("--output_json", type=str, default="nudenet_results.json", help="Path to save the detection results JSON.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection (default: 0.5).")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"], help="Image file extensions to look for.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # Initialize NudeDetector
    print("Initializing NudeNet Detector...")
    detector = NudeDetector()

    # Get all leaf directories
    leaf_dirs = get_leaf_directories(args.input_dir)
    print(f"Found {len(leaf_dirs)} leaf directories to process.")

    all_results = {}

    for leaf_dir in tqdm(leaf_dirs, desc="Processing directories"):
        folder_results = detect_images_in_folder(detector, leaf_dir, args.extensions, args.threshold)
        
        if folder_results:
            # Use relative path from input_dir as key to keep structure clean
            rel_path = os.path.relpath(leaf_dir, args.input_dir)
            all_results[rel_path] = folder_results

    # Save results
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Detection complete. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()