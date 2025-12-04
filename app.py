import os
os.environ['GLOG_minloglevel'] = '3'
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
from inference import SignLanguageModel, extract_keypoints, draw_styled_landmarks, is_idle
import time
import base64
import numpy as np
import threading


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-fallback-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# ==================== CONFIGURATION ====================
MODEL_PATH = "ArabSignModel.pth"
LABELS_CSV = "01_test.csv"
CAMERA_INDEX = 0


# Test video paths (UPDATE THESE WITH YOUR ACTUAL PATHS!)
TEST_VIDEOS = {
    'test1': r"C:\Users\wizmo\Desktop\test1.mp4",
    'test2': r"C:\Users\wizmo\Desktop\test2.mp4",
    'test3': r"C:\Users\wizmo\Desktop\test3.mp4",
}


SEQUENCE_LENGTH = 80
CONFIDENCE_THRESHOLD = 0.30
PAUSE_THRESHOLD = 18
MIN_SIGN_FRAMES = 12


# ==================== GLOBAL STATE ====================
sign_model = None
model_lock = threading.Lock()
detected_words = {'live': [], 'video': []}
show_skeleton = True
camera = None
camera_active = False
current_mode = 'live'
video_processing = False
current_test_video = None

# User sessions for mobile camera
user_sessions = {}
last_process_time = {}


def get_user_session(sid):
    """Get or create user session"""
    if sid not in user_sessions:
        user_sessions[sid] = {
            'words_mobile': [],
            'show_skeleton': True,
            'mobile_buffer': [],
            'mobile_idle': 0,
            'mobile_signing': 0,
            'mobile_active': False,
            'holistic': None,
            'last_prediction': -999,
            'frame_count': 0
        }
    return user_sessions[sid]


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
    signing_counter = 0
    frame_count = 0
    last_prediction_frame = -999
    
    print("✅ Live camera started")
    
    try:
        while camera_active:
            success, frame = cam.read()
            if not success:
                continue
            
            frame_count += 1
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
                signing_counter = 0
                
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
                if (idle_counter >= PAUSE_THRESHOLD and 
                    len(keypoints_buffer) >= MIN_SIGN_FRAMES and
                    frame_count - last_prediction_frame > 30):
                    
                    pose_seq = [kp[0] for kp in keypoints_buffer]
                    lh_seq = [kp[1] for kp in keypoints_buffer]
                    rh_seq = [kp[2] for kp in keypoints_buffer]
                    
                    with model_lock:
                        try:
                            word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                            
                            if confidence > CONFIDENCE_THRESHOLD:
                                if not detected_words['live'] or detected_words['live'][-1] != word:
                                    detected_words['live'].append(word)
                                    print(f"✅ Live: {word} ({confidence:.2f}) [Buffer: {len(keypoints_buffer)} frames]")
                                
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                            (0, 255, 0), 12)
                            else:
                                print(f"⚠️ Low confidence: {confidence:.2f} [Buffer: {len(keypoints_buffer)} frames]")
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                            (0, 0, 255), 12)
                        except Exception as e:
                            print(f"⚠️ Prediction error: {e}")
                    
                    keypoints_buffer = []
                    idle_counter = 0
                    last_prediction_frame = frame_count
            else:
                signing_counter += 1
                
                if signing_counter >= 3:
                    idle_counter = 0
                
                keypoints_buffer.append((pose, lh, rh))
                
                if len(keypoints_buffer) > 150:
                    keypoints_buffer.pop(0)
                
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
    signing_counter = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_prediction_frame = -999
    
    # Clear video words before starting
    detected_words['video'] = []
    
    print(f"✅ Processing video: {current_test_video} ({total_frames} frames)")
    
    try:
        while video_processing:
            success, frame = cap.read()
            
            # Stop when video ends (no looping)
            if not success:
                print(f"✅ Video finished: {current_test_video}")
                video_processing = False
                break
            
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
                signing_counter = 0
                
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
                if (idle_counter >= PAUSE_THRESHOLD and 
                    len(keypoints_buffer) >= MIN_SIGN_FRAMES and
                    frame_count - last_prediction_frame > 30):
                    
                    pose_seq = [kp[0] for kp in keypoints_buffer]
                    lh_seq = [kp[1] for kp in keypoints_buffer]
                    rh_seq = [kp[2] for kp in keypoints_buffer]
                    
                    with model_lock:
                        try:
                            word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                            
                            if confidence > CONFIDENCE_THRESHOLD:
                                if not detected_words['video'] or detected_words['video'][-1] != word:
                                    detected_words['video'].append(word)
                                    print(f"✅ Video: {word} ({confidence:.2f}) [Buffer: {len(keypoints_buffer)} frames]")
                                
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                            (0, 255, 0), 12)
                            else:
                                print(f"⚠️ Low confidence: {confidence:.2f} [Buffer: {len(keypoints_buffer)} frames]")
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                            (0, 0, 255), 12)
                        except Exception as e:
                            print(f"⚠️ Prediction error: {e}")
                    
                    keypoints_buffer = []
                    idle_counter = 0
                    last_prediction_frame = frame_count
            else:
                signing_counter += 1
                
                if signing_counter >= 3:
                    idle_counter = 0
                
                keypoints_buffer.append((pose, lh, rh))
                
                if len(keypoints_buffer) > 150:
                    keypoints_buffer.pop(0)
                
                cv2.putText(frame, "SIGNING", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Frames: {len(keypoints_buffer)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.circle(frame, (frame.shape[1] - 40, 40), 15, (0, 0, 255), -1)
            
            # Progress bar
            cv2.putText(frame, f"{current_test_video.upper()}: {frame_count}/{total_frames}", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.016)  # ~60 FPS (faster playback)
    
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
            # Show placeholder when video is off
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
    """AI refinement with Arabic sign language context - optimized for khutbah"""
    try:
        from groq import Groq
        
        data = request.json
        mode = data.get('mode', 'live')
        
        # Get words
        if mode == 'mobile':
            sid = request.headers.get('X-Session-ID')
            if sid and sid in user_sessions:
                words_list = user_sessions[sid]['words_mobile']
            else:
                words_list = []
        else:
            words_list = detected_words.get(mode, [])
        
        if not words_list:
            return jsonify({'error': 'لا توجد كلمات'}), 400
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return jsonify({'error': 'GROQ_API_KEY not set'}), 400
        
        client = Groq(api_key=api_key)
        words = " ".join(words_list)
        
        # IMPROVED PROMPT for Arabic
        prompt = f"""أنت خبير في ترجمة لغة الإشارة العربية إلى نص عربي فصيح.

السياق: الإشارات من خطبة جمعة تتضمن:
- دعاء وتوحيد
- صفات الله تعالى
- عبارات دينية

قواعد الترجمة:
1. لغة الإشارة تحذف "ال" التعريف - أضفها للوضوح
2. لغة الإشارة تحذف حروف الجر - أضفها حسب الحاجة
3. صحح تصريف الأفعال (مثال: "انا حب" → "أنا أحب")
4. احتفظ بالمعنى الأصلي بدون إضافات غير ضرورية
5. اكتب جملاً عربية سليمة نحوياً

أمثلة من الخطبة:
الإشارات: "الله قبول عمل تمني الله فائدة"
الترجمة: "نسأل الله قبول العمل وأن يرزقنا الفائدة."

الإشارات: "الله كريم الله رزق الله غني"
الترجمة: "الله كريم رزاق غني."

الإشارات: "لا شرك الله عبادة الله واحد"
الترجمة: "لا شريك لله العبادة لله وحده."

الإشارات: "اسم الله حمد الله سلام عليكم رحمة الله بركة"
الترجمة: "بسم الله. الحمد لله. السلام عليكم ورحمة الله وبركاته."

الإشارات المكتشفة: "{words}"

الترجمة العربية الفصيحة (فقط النص المترجم بدون شرح):"""

   
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=150,
            top_p=0.9,
        )
        
        refined = completion.choices[0].message.content.strip()
        
        # Clean up the output
        if ':' in refined or '：' in refined:
            refined = refined.split(':', 1)[-1].split('：', 1)[-1].strip()
        
        prefixes_to_remove = ['الترجمة:', 'الترجمة :', 'النص:', 'الجملة:']
        for prefix in prefixes_to_remove:
            if refined.startswith(prefix):
                refined = refined[len(prefix):].strip()
        
        return jsonify({'refined_text': refined})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'خطأ في المعالجة: {str(e)}'}), 500


# ==================== SOCKET.IO EVENTS ====================
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    get_user_session(sid)
    print(f"✅ Mobile user connected: {sid[:8]}")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in user_sessions:
        session = user_sessions[sid]
        if session['holistic']:
            session['holistic'].close()
        del user_sessions[sid]
    if sid in last_process_time:
        del last_process_time[sid]
    print(f"❌ Mobile user disconnected: {sid[:8]}")


@socketio.on('mobile_start')
def handle_mobile_start():
    sid = request.sid
    session = get_user_session(sid)
    session['mobile_active'] = True
    session['words_mobile'] = []
    session['mobile_buffer'] = []
    session['mobile_idle'] = 0
    session['mobile_signing'] = 0
    session['last_prediction'] = -999
    session['frame_count'] = 0
    
    # Create MediaPipe instance ONCE per session
    if not session['holistic']:
        mp_holistic = mp.solutions.holistic
        session['holistic'] = mp_holistic.Holistic(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
    
    print(f"📱 Mobile camera started for user: {sid[:8]}")
    emit('mobile_started')


@socketio.on('mobile_stop')
def handle_mobile_stop():
    sid = request.sid
    session = get_user_session(sid)
    session['mobile_active'] = False
    session['mobile_buffer'] = []
    session['mobile_idle'] = 0
    session['mobile_signing'] = 0
    
    print(f"📱 Mobile camera stopped for user: {sid[:8]}")
    emit('mobile_stopped')


@socketio.on('mobile_frame')
def handle_mobile_frame(data):
    """Process frame from mobile camera - OPTIMIZED FOR NGROK"""
    sid = request.sid
    session = get_user_session(sid)
    
    if not session['mobile_active'] or not session['holistic']:
        return
    
    # OPTIMIZED: More aggressive frame skipping for mobile/ngrok
    current_time = time.time()
    if sid in last_process_time:
        if current_time - last_process_time[sid] < 0.12:
            return
    last_process_time[sid] = current_time
    
    model = load_model()
    session['frame_count'] += 1
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Fix mirror issue
        frame = cv2.flip(frame, 1)
        
        # OPTIMIZED: Resize frame for faster processing
        frame = cv2.resize(frame, (320, 240))
        
        holistic = session['holistic']
        
        # MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        pose = keypoints[:99]
        lh = keypoints[99:162]
        rh = keypoints[162:225]
        
        # Idle detection
        is_person_idle = is_idle(results)
        
        if is_person_idle:
            session['mobile_idle'] += 1
            session['mobile_signing'] = 0
            
            # Trigger prediction
            if (session['mobile_idle'] >= PAUSE_THRESHOLD and 
                len(session['mobile_buffer']) >= MIN_SIGN_FRAMES and
                session['frame_count'] - session['last_prediction'] > 20):
                
                pose_seq = [kp[0] for kp in session['mobile_buffer']]
                lh_seq = [kp[1] for kp in session['mobile_buffer']]
                rh_seq = [kp[2] for kp in session['mobile_buffer']]
                
                with model_lock:
                    try:
                        word, confidence = model.predict(pose_seq, lh_seq, rh_seq, SEQUENCE_LENGTH)
                        
                        # Lower confidence threshold for mobile
                        if confidence > (CONFIDENCE_THRESHOLD - 0.05):
                            if not session['words_mobile'] or session['words_mobile'][-1] != word:
                                session['words_mobile'].append(word)
                                print(f"✅ Mobile {sid[:8]}: {word} ({confidence:.2f}) [Buffer: {len(session['mobile_buffer'])} frames]")
                                socketio.emit('mobile_words', {'words': session['words_mobile']}, room=sid)
                        else:
                            print(f"⚠️ Mobile low confidence: {confidence:.2f}")
                        
                    except Exception as e:
                        print(f"⚠️ Mobile prediction: {str(e)[:50]}")
                
                session['mobile_buffer'] = []
                session['mobile_idle'] = 0
                session['last_prediction'] = session['frame_count']
        else:
            session['mobile_signing'] += 1
            
            if session['mobile_signing'] >= 3:
                session['mobile_idle'] = 0
            
            session['mobile_buffer'].append((pose, lh, rh))
            
            if len(session['mobile_buffer']) > 120:
                session['mobile_buffer'].pop(0)
        
    except Exception as e:
        print(f"⚠️ Frame error: {str(e)[:80]}")


@socketio.on('get_mobile_words')
def handle_get_mobile_words():
    sid = request.sid
    session = get_user_session(sid)
    emit('mobile_words', {'words': session['words_mobile']})


@socketio.on('clear_mobile_words')
def handle_clear_mobile_words():
    sid = request.sid
    session = get_user_session(sid)
    session['words_mobile'] = []
    session['mobile_buffer'] = []
    session['mobile_idle'] = 0
    session['mobile_signing'] = 0
    emit('mobile_words', {'words': []})
    print(f"🗑️ Cleared history for {sid[:8]}")
# ==================== TTS ROUTES ====================
@app.route('/api/tts/word', methods=['POST'])
def tts_word():
    """Generate TTS for a single word"""
    try:
        from gtts import gTTS
        import io
        
        data = request.json
        word = data.get('word', '')
        
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        # Generate TTS
        tts = gTTS(text=word, lang='ar', slow=False)
        
        # Save to BytesIO buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return Response(audio_buffer.getvalue(), mimetype='audio/mpeg')
    
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/sentence', methods=['POST'])
def tts_sentence():
    """Generate TTS for full sentence (refined text)"""
    try:
        from gtts import gTTS
        import io
        
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate TTS for Arabic text
        tts = gTTS(text=text, lang='ar', slow=False)
        
        # Save to BytesIO buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return Response(audio_buffer.getvalue(), mimetype='audio/mpeg')
    
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/all', methods=['POST'])
def tts_all():
    """Generate TTS for all detected words"""
    try:
        from gtts import gTTS
        import io
        
        data = request.json
        mode = data.get('mode', 'live')
        
        # Get words
        if mode == 'mobile':
            sid = request.headers.get('X-Session-ID')
            if sid and sid in user_sessions:
                words_list = user_sessions[sid]['words_mobile']
            else:
                words_list = []
        else:
            words_list = detected_words.get(mode, [])
        
        if not words_list:
            return jsonify({'error': 'No words to speak'}), 400
        
        # Join all words
        text = ' '.join(words_list)
        
        # Generate TTS
        tts = gTTS(text=text, lang='ar', slow=False)
        
        # Save to BytesIO buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return Response(audio_buffer.getvalue(), mimetype='audio/mpeg')
    
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ASL TRANSLATOR - COMPLETE VERSION + MOBILE")
    print("="*70)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📋 Labels: {LABELS_CSV}")
    print(f"📹 Camera: {CAMERA_INDEX}")
    print(f"🎬 Test Videos: {len(TEST_VIDEOS)}")
    print(f"🌐 Laptop: http://127.0.0.1:5000")
    print(f"📱 Mobile: Run 'ngrok http 5000' in another terminal")
    print(f"⚡ Optimized: PAUSE={PAUSE_THRESHOLD}, MIN_FRAMES={MIN_SIGN_FRAMES}")
    print("="*70 + "\n")
    
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
