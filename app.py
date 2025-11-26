"""
ASL Translator - Complete Deployable Version
Live Camera + Video Test Modes
"""
import os
os.environ['GLOG_minloglevel'] = '3'

from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from inference import SignLanguageModel, extract_keypoints, is_idle
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================== CONFIGURATION ====================
MODEL_PATH = "ArabSignModel.pth"
LABELS_CSV = "01_test.csv"

# Test video paths (you'll need to upload these or remove this feature)
TEST_VIDEOS = {
    'test1': 'test_videos/test1.mp4',
    'test2': 'test_videos/test2.mp4',
    'test3': 'test_videos/test3.mp4',
    'test4': 'test_videos/test4.mp4',
    'test5': 'test_videos/test5.mp4',
}

SEQUENCE_LENGTH = 80
CONFIDENCE_THRESHOLD = 0.30
PAUSE_THRESHOLD = 15
MIN_SIGN_FRAMES = 20

# ==================== GLOBAL STATE ====================
sign_model = None
user_sessions = {}
mp_holistic = mp.solutions.holistic

def load_model():
    global sign_model
    if sign_model is None:
        print("🔄 Loading model...")
        sign_model = SignLanguageModel(MODEL_PATH, LABELS_CSV)
    return sign_model

class UserSession:
    def __init__(self):
        self.detected_words = []
        self.keypoints_buffer = []
        self.idle_counter = 0
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.show_skeleton = True
        self.mode = 'live'  # 'live' or 'video'
    
    def cleanup(self):
        if self.holistic:
            self.holistic.close()

# ==================== SOCKETIO EVENTS ====================
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    user_sessions[session_id] = UserSession()
    print(f"✅ User connected: {session_id}")
    emit('connected', {'status': 'success'})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in user_sessions:
        user_sessions[session_id].cleanup()
        del user_sessions[session_id]
    print(f"👋 User disconnected: {session_id}")

@socketio.on('process_frame')
def handle_frame(data):
    """Process frame from client's camera"""
    session_id = request.sid
    
    if session_id not in user_sessions:
        return
    
    session = user_sessions[session_id]
    model = load_model()
    
    try:
        # Decode frame
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = session.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        pose = keypoints[:99]
        lh = keypoints[99:162]
        rh = keypoints[162:225]
        
        # Idle detection
        is_person_idle = is_idle(results)
        
        response = {
            'idle': is_person_idle,
            'idle_counter': session.idle_counter,
            'buffer_size': len(session.keypoints_buffer),
            'detected_word': None,
            'confidence': 0,
            'flash': None
        }
        
        if is_person_idle:
            session.idle_counter += 1
            
            if session.idle_counter >= PAUSE_THRESHOLD and len(session.keypoints_buffer) >= MIN_SIGN_FRAMES:
                pose_seq = [kp[0] for kp in session.keypoints_buffer]
                lh_seq = [kp[1] for kp in session.keypoints_buffer]
                rh_seq = [kp[2] for kp in session.keypoints_buffer]
                
                try:
                    word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        if not session.detected_words or session.detected_words[-1] != word:
                            session.detected_words.append(word)
                            print(f"✅ {session_id}: {word} ({confidence:.2f})")
                        
                        response['detected_word'] = word
                        response['confidence'] = confidence
                        response['flash'] = 'green'
                        response['all_words'] = session.detected_words
                    else:
                        response['flash'] = 'red'
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
                
                session.keypoints_buffer = []
                session.idle_counter = 0
        else:
            session.idle_counter = 0
            session.keypoints_buffer.append((pose, lh, rh))
        
        emit('frame_result', response)
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        emit('error', {'message': str(e)})

@socketio.on('get_words')
def handle_get_words():
    session_id = request.sid
    if session_id in user_sessions:
        emit('words_update', {'words': user_sessions[session_id].detected_words})

@socketio.on('clear_words')
def handle_clear_words():
    session_id = request.sid
    if session_id in user_sessions:
        user_sessions[session_id].detected_words = []
        emit('words_cleared', {'status': 'success'})

@socketio.on('toggle_skeleton')
def handle_toggle_skeleton():
    session_id = request.sid
    if session_id in user_sessions:
        user_sessions[session_id].show_skeleton = not user_sessions[session_id].show_skeleton
        emit('skeleton_toggled', {'show': user_sessions[session_id].show_skeleton})

# ==================== HTTP ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': sign_model is not None}

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ASL TRANSLATOR - PRODUCTION")
    print("="*70)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📋 Labels: {LABELS_CSV}")
    print("="*70 + "\n")
    
    load_model()
    
    # Use PORT environment variable (required by Render)
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
