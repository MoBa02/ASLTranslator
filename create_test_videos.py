"""
Create 3 test videos by combining dataset videos
Run this ONCE to generate test1.mp4, test2.mp4, test3.mp4
"""

import cv2
import os
import pandas as pd

# ==================== CONFIGURATION ====================
BASE_DIR = r"C:\Users\wizmo\OneDrive\Desktop\SDP-python\ArabSign\Color\01\test"
OUTPUT_DIR = r"C:\Users\wizmo\Desktop"
CSV_FILE = "01_test.csv"

# ==================== TEST DEFINITIONS ====================
TESTS = {
    'test1': {
        'labels': [36, 37],
        'description': 'دعاء الخاتمة',
        'expected_llm': 'نسأل الله قبول العمل وأن يرزقنا الفائدة.'
    },
    'test2': {
        'labels': [13, 14, 17],
        'description': 'صفات الله',
        'expected_llm': 'الله كريم رزاق غني.'
    },
    'test3': {
        'labels': [9, 12],
        'description': 'التوحيد',
        'expected_llm': 'لا شرك بالله. العبادة لله الواحد.'
    }
}

# ==================== FUNCTIONS ====================

def get_first_video_from_label(base_dir, label_id):
    """Get the first video file for a given label"""
    folder_name = f"{label_id:04d}"  # e.g., 0036
    folder_path = os.path.join(base_dir, folder_name)

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return None

    videos = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])

    if not videos:
        print(f"❌ No videos in folder: {folder_path}")
        return None

    video_path = os.path.join(folder_path, videos[0])
    print(f"   ✅ Found: {videos[0]}")
    return video_path

def combine_videos(video_paths, output_path):
    """Combine multiple videos into one"""
    if not video_paths:
        print("❌ No videos to combine!")
        return False

    # Read first video to get dimensions and fps
    first_cap = cv2.VideoCapture(video_paths[0])
    fps = int(first_cap.get(cv2.CAP_PROP_FPS))
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    print(f"   Video settings: {width}x{height} @ {fps}fps")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, video_path in enumerate(video_paths, 1):
        print(f"   [{i}/{len(video_paths)}] Adding: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        print(f"        → {frame_count} frames added")
        cap.release()

    out.release()
    print(f"   ✅ Created: {os.path.basename(output_path)}\n")
    return True

# ==================== MAIN ====================

def create_test_videos():
    """Create all test videos"""

    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        return

    # Read CSV for label names
    df = pd.read_csv(CSV_FILE, header=None, 
                     names=['type', 'id', 'path', 'frames', 'unknown', 'full_text', 'sign_words'])

    print("="*80)
    print("🎬 CREATING TEST VIDEOS FROM DATASET")
    print("="*80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0

    for test_name, test_data in TESTS.items():
        print("="*80)
        print(f"📹 {test_name.upper()}: {test_data['description']}")
        print("="*80)
        print(f"Expected LLM output: {test_data['expected_llm']}\n")

        video_paths = []

        for label_id in test_data['labels']:
            label_text = df[df['id'] == label_id]['sign_words'].iloc[0]
            print(f"Label {label_id}: {label_text}")

            video_path = get_first_video_from_label(BASE_DIR, label_id)
            if video_path:
                video_paths.append(video_path)

        # Combine videos
        if video_paths:
            output_path = os.path.join(OUTPUT_DIR, f"{test_name}.mp4")
            if combine_videos(video_paths, output_path):
                success_count += 1
        else:
            print(f"❌ No videos found for {test_name}\n")

    print("="*80)
    print(f"✅ DONE! Created {success_count}/3 test videos")
    print("="*80)

    if success_count > 0:
        print(f"""
📝 Update your app.py with:

TEST_VIDEOS = {{
    'test1': r"{os.path.join(OUTPUT_DIR, 'test1.mp4')}",
    'test2': r"{os.path.join(OUTPUT_DIR, 'test2.mp4')}",
    'test3': r"{os.path.join(OUTPUT_DIR, 'test3.mp4')}",
}}
""")

if __name__ == "__main__":
    create_test_videos()