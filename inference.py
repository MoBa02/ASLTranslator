"""
Sign Language Inference Module
Extracted from your notebook - cleaned and optimized
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import os

# Suppress logs
os.environ['GLOG_minloglevel'] = '3'

# ==================== PREPROCESSING ====================
def interpolate_seq(arr, target_len):
    T, D = arr.shape
    if T == 0:
        return np.zeros((target_len, D), dtype=arr.dtype)
    if T == target_len:
        return arr
    orig_idx = np.linspace(0, T-1, num=T)
    tgt_idx = np.linspace(0, T-1, num=target_len)
    return np.stack([np.interp(tgt_idx, orig_idx, arr[:, d]) for d in range(D)], axis=1)

def normalize_seq(arr):
    T, D = arr.shape
    C = 3 if D % 3 == 0 else 2
    pts = arr.reshape(T, -1, C)
    hip_mid = (pts[:, 11, :2] + pts[:, 12, :2]) / 2.0
    pts[..., :2] -= hip_mid[:, None, :]
    torso = np.linalg.norm(pts[:, 11, :2] - pts[:, 12, :2], axis=1)
    scale = np.maximum(torso, 1e-6)
    pts[..., :2] /= scale[:, None, None]
    return pts.reshape(T, D)

def preprocess_for_inference(pose_seq, lh_seq, rh_seq, f_avg=80):
    """Preprocess exactly like training: interpolate + normalize + concatenate."""
    pose_arr = np.vstack(pose_seq)
    lh_arr = np.vstack(lh_seq)
    rh_arr = np.vstack(rh_seq)
    
    pose_i = interpolate_seq(pose_arr, f_avg)
    lh_i = interpolate_seq(lh_arr, f_avg)
    rh_i = interpolate_seq(rh_arr, f_avg)
    
    pose_n = normalize_seq(pose_i)
    lh_n = normalize_seq(lh_i)
    rh_n = normalize_seq(rh_i)
    
    combined = np.concatenate([pose_n, lh_n, rh_n], axis=1)
    return np.expand_dims(combined, axis=0)

# ==================== KEYPOINT EXTRACTION ====================
def extract_keypoints(results):
    """Extract pose + lh + rh (225 features)."""
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(99)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw pose/hand landmarks with your custom colors."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def is_idle(results, idle_threshold_y=0.9):
    """Check if hands are down (idle position)."""
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return True
    
    if results.left_hand_landmarks:
        lh_avg_y = np.mean([lm.y for lm in results.left_hand_landmarks.landmark])
        if lh_avg_y > idle_threshold_y:
            return True
    
    if results.right_hand_landmarks:
        rh_avg_y = np.mean([lm.y for lm in results.right_hand_landmarks.landmark])
        if rh_avg_y > idle_threshold_y:
            return True
    
    return False

# ==================== MODEL ====================
class BiLSTMModel(torch.nn.Module):
    def __init__(self, input_size=225, hidden_size=64, num_layers=2, num_classes=50, dropout=0.2):
        super().__init__()
        self.bilstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.drop = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size * 2, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==================== MODEL LOADER ====================
class SignLanguageModel:
    def __init__(self, model_path, labels_csv):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = BiLSTMModel(input_size=225)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load labels
        labels_df = pd.read_csv(labels_csv)
        self.label_map = {}
        seen_ids = set()
        
        for idx, row in labels_df.iterrows():
            sentence_id = row.iloc[1] - 1
            if sentence_id not in seen_ids:
                arabic_text = row.iloc[5]
                self.label_map[sentence_id] = arabic_text
                seen_ids.add(sentence_id)
        
        print(f"✅ Model loaded: {len(self.label_map)} classes on {self.device}")
    
    def predict(self, pose_seq, lh_seq, rh_seq, sequence_length=80):
        """Run prediction on keypoint sequences."""
        input_data = preprocess_for_inference(pose_seq, lh_seq, rh_seq, f_avg=sequence_length)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            
            word = self.label_map.get(predicted_class.item(), "Unknown")
            return word, confidence.item()
