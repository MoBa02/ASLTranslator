"""
ASL Translator - Complete Version with All Fixes
"""
import os
os.environ['GLOG_minloglevel'] = '3'

from flask import Flask, Response, render_template, jsonify, request
import cv2
import mediapipe as mp
from inference import SignLanguageModel, extract_keypoints, draw_styled_landmarks, is_idle
import time

app = Flask(__name__)

# ==================== CONFIGURATION ====================
MODEL_PATH = "ArabSignModel.pth"
LABELS_CSV = "01_test.csv"
CAMERA_INDEX = 0

# Test video paths (UPDATE THESE WITH YOUR ACTUAL PATHS!)
TEST_VIDEOS = {
    'test1': r"C:\path\to\test1.mp4",
    'test2': r"C:\path\to\test2.mp4",
    'test3': r"C:\path\to\test3.mp4",
    'test4': r"C:\path\to\test4.mp4",
    'test5': r"C:\path\to\test5.mp4",
}

SEQUENCE_LENGTH = 80
CONFIDENCE_THRESHOLD = 0.30
PAUSE_THRESHOLD = 15
MIN_SIGN_FRAMES = 20

# ==================== GLOBAL STATE ====================
sign_model = None
detected_words = {'live': [], 'video': []}
show_skeleton = True
camera = None
camera_active = False
current_mode = 'live'
video_processing = False
current_test_video = None

def get_camera():
    """Singleton camera instance"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera

def release_camera():
    """Release camera"""
    global camera, camera_active
    if camera:
        camera.release()
        camera = None
    camera_active = False

def load_model():
    """Lazy load model"""
    global sign_model
    if sign_model is None:
        print("🔄 Loading model...")
        sign_model = SignLanguageModel(MODEL_PATH, LABELS_CSV)
    return sign_model

# ==================== LIVE CAMERA GENERATOR ====================
def generate_live_frames():
    global detected_words, show_skeleton, camera_active
    
    if not camera_active:
        return
    
    cam = get_camera()
    if not cam or not cam.isOpened():
        print("❌ Camera failed")
        camera_active = False
        return
    
    model = load_model()
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    keypoints_buffer = []
    idle_counter = 0
    
    print("✅ Live camera started")
    
    try:
        while camera_active:
            success, frame = cam.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Draw skeleton
            if show_skeleton:
                draw_styled_landmarks(frame, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            pose = keypoints[:99]
            lh = keypoints[99:162]
            rh = keypoints[162:225]
            
            # Idle detection
            is_person_idle = is_idle(results)
            
            if is_person_idle:
                idle_counter += 1
                
                cv2.putText(frame, "IDLE", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # Progress circle
                radius = 20
                center = (frame.shape[1] - 40, 40)
                progress = min(idle_counter / PAUSE_THRESHOLD, 1.0)
                cv2.circle(frame, center, radius, (50, 50, 50), -1)
                cv2.ellipse(frame, center, (radius, radius), -90, 0, 
                           int(360 * progress), (0, 255, 255), 3)
                
                # Trigger prediction
                if idle_counter >= PAUSE_THRESHOLD and len(keypoints_buffer) >= MIN_SIGN_FRAMES:
                    pose_seq = [kp[0] for kp in keypoints_buffer]
                    lh_seq = [kp[1] for kp in keypoints_buffer]
                    rh_seq = [kp[2] for kp in keypoints_buffer]
                    
                    try:
                        word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            if not detected_words['live'] or detected_words['live'][-1] != word:
                                detected_words['live'].append(word)
                                print(f"✅ Live: {word} ({confidence:.2f})")
                            
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                        (0, 255, 0), 12)
                        else:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                        (0, 0, 255), 12)
                    except Exception as e:
                        print(f"⚠️ Prediction error: {e}")
                    
                    keypoints_buffer = []
                    idle_counter = 0
            else:
                idle_counter = 0
                keypoints_buffer.append((pose, lh, rh))
                
                cv2.putText(frame, "SIGNING", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Frames: {len(keypoints_buffer)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.circle(frame, (frame.shape[1] - 40, 40), 15, (0, 0, 255), -1)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except GeneratorExit:
        print("Live stream closed")
    finally:
        holistic.close()

# ==================== VIDEO FILE GENERATOR ====================
def generate_video_frames():
    global detected_words, show_skeleton, current_test_video, video_processing
    
    if not current_test_video or current_test_video not in TEST_VIDEOS:
        video_processing = False
        return
    
    video_path = TEST_VIDEOS[current_test_video]
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        video_processing = False
        return
    
    model = load_model()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        video_processing = False
        return
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    keypoints_buffer = []
    idle_counter = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clear video words before starting
    detected_words['video'] = []
    
    print(f"✅ Processing video: {current_test_video} ({total_frames} frames)")
    
    try:
        while video_processing:
            success, frame = cap.read()
            if not success:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Draw skeleton
            if show_skeleton:
                draw_styled_landmarks(frame, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            pose = keypoints[:99]
            lh = keypoints[99:162]
            rh = keypoints[162:225]
            
            # Idle detection
            is_person_idle = is_idle(results)
            
            if is_person_idle:
                idle_counter += 1
                
                cv2.putText(frame, "IDLE", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # Progress
                radius = 20
                center = (frame.shape[1] - 40, 40)
                progress = min(idle_counter / PAUSE_THRESHOLD, 1.0)
                cv2.circle(frame, center, radius, (50, 50, 50), -1)
                cv2.ellipse(frame, center, (radius, radius), -90, 0, 
                           int(360 * progress), (0, 255, 255), 3)
                
                # Trigger prediction
                if idle_counter >= PAUSE_THRESHOLD and len(keypoints_buffer) >= MIN_SIGN_FRAMES:
                    pose_seq = [kp[0] for kp in keypoints_buffer]
                    lh_seq = [kp[1] for kp in keypoints_buffer]
                    rh_seq = [kp[2] for kp in keypoints_buffer]
                    
                    try:
                        word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            if not detected_words['video'] or detected_words['video'][-1] != word:
                                detected_words['video'].append(word)
                                print(f"✅ Video: {word} ({confidence:.2f})")
                            
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                        (0, 255, 0), 12)
                        else:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                        (0, 0, 255), 12)
                    except Exception as e:
                        print(f"⚠️ Prediction error: {e}")
                    
                    keypoints_buffer = []
                    idle_counter = 0
            else:
                idle_counter = 0
                keypoints_buffer.append((pose, lh, rh))
                
                cv2.putText(frame, "SIGNING", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.circle(frame, (frame.shape[1] - 40, 40), 15, (0, 0, 255), -1)
            
            # Progress bar
            cv2.putText(frame, f"{current_test_video.upper()}: {frame_count}/{total_frames}", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
    
    except GeneratorExit:
        print("Video stream closed")
    finally:
        cap.release()
        holistic.close()
        video_processing = False

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global current_mode, camera_active, video_processing
    
    if current_mode == 'live':
        if camera_active:
            return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            # Return placeholder when camera off
            import numpy as np
            def placeholder_gen():
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Off - Click Start", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                ret, buf = cv2.imencode('.jpg', placeholder)
                while True:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            return Response(placeholder_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    elif current_mode == 'video':
        if video_processing:
            return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            # Return placeholder
            import numpy as np
            def placeholder_gen():
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Select a Test Video", (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                ret, buf = cv2.imencode('.jpg', placeholder)
                while True:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            return Response(placeholder_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/toggle', methods=['POST'])
def toggle_camera():
    global camera_active, current_mode, video_processing
    
    # Make sure we're in live mode
    current_mode = 'live'
    video_processing = False  # Stop any video
    
    if camera_active:
        release_camera()
        return jsonify({'status': 'stopped', 'active': False})
    else:
        camera_active = True
        return jsonify({'status': 'started', 'active': True})

@app.route('/api/video/start', methods=['POST'])
def start_video_test():
    global video_processing, current_test_video, current_mode, camera_active
    
    data = request.json
    test_name = data.get('test', 'test1')
    
    if test_name not in TEST_VIDEOS:
        return jsonify({'error': 'Invalid test video'}), 400
    
    # Stop camera if running
    if camera_active:
        release_camera()
    
    current_mode = 'video'
    current_test_video = test_name
    video_processing = True
    
    return jsonify({'status': 'started', 'test': test_name})

@app.route('/api/video/stop', methods=['POST'])
def stop_video_test():
    global video_processing
    video_processing = False
    return jsonify({'status': 'stopped'})

@app.route('/api/words')
def get_words():
    mode = request.args.get('mode', 'live')
    return jsonify({'words': detected_words.get(mode, [])})

@app.route('/api/clear')
def clear_words():
    mode = request.args.get('mode', 'live')
    detected_words[mode] = []
    return jsonify({'status': 'cleared'})

@app.route('/api/toggle_skeleton')
def toggle_skeleton():
    global show_skeleton
    show_skeleton = not show_skeleton
    return jsonify({'show_skeleton': show_skeleton})

@app.route('/api/refine', methods=['POST'])
def refine_text():
    """AI sentence refinement"""
    try:
        import google.generativeai as genai
        
        data = request.json
        mode = data.get('mode', 'live')
        words_list = detected_words.get(mode, [])
        
        if not words_list:
            return jsonify({'error': 'لا توجد كلمات'}), 400
        
        api_key = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
        if api_key == 'YOUR_API_KEY_HERE':
            return jsonify({'error': 'Set GEMINI_API_KEY environment variable'}), 400
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        words = " ".join(words_list)
        prompt = f"""أنت مترجم للغة الإشارة العربية. حوّل هذه الكلمات المنفصلة إلى جملة عربية طبيعية وصحيحة نحوياً.

الكلمات: {words}

اكتب فقط الجملة النهائية بالعربية، بدون أي شرح."""
        
        response = model.generate_content(prompt)
        return jsonify({'refined_text': response.text})
    
    except ImportError:
        return jsonify({'error': 'Install: pip install google-generativeai'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ASL TRANSLATOR - COMPLETE VERSION")
    print("="*70)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📋 Labels: {LABELS_CSV}")
    print(f"📹 Camera: {CAMERA_INDEX}")
    print(f"🎬 Test Videos: {len(TEST_VIDEOS)}")
    print(f"🌐 Open: http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
