import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

ACTIONS = np.array(['apa', 'apa_kabar', 'bagaimana', 'baik', 'belajar', 'berapa', 'berdiri', 
    'bingung', 'dia', 'dimana', 'duduk', 'halo', 'kalian', 'kami', 'kamu', 
    'kapan', 'kemana', 'kita', 'makan', 'mandi', 'marah', 'melihat', 'membaca', 
    'menulis', 'mereka', 'minum', 'pendek', 'ramah', 'sabar', 'saya', 'sedih', 
    'selamat_malam', 'selamat_pagi', 'selamat_siang', 'selamat_sore', 'senang', 
    'siapa', 'terima_kasih', 'tidur', 'tinggi'
])

# Face selection for Lightweight performance
SELECTED_FACE_IDS = [
    # Lips (For mouthing/shape)
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 415,
    # Eyebrows
    46, 52, 53, 55, 65, 70, 105, 107, 276, 282, 283, 285, 295, 300, 334, 336,
    # Left Cheek Zone
    50, 118, 123, 137, 205, 206, 207, 212, 214, 216,
    # Right Cheek Zone
    280, 347, 352, 366, 425, 426, 427, 432, 434, 436
]

SEQUENCE_LENGTH = 30 
DATA_PATH = 'features'
VIDEO_PATH = 'videos'
DEBUG_PATH = 'debug_videos'

# Create necessary directories
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
    os.makedirs(os.path.join(VIDEO_PATH, action), exist_ok=True)
os.makedirs(DEBUG_PATH, exist_ok=True)

def extract_and_normalize_keypoints(results):
    """Applies Shoulder Normalization to handle translation and distance/scale"""
    cx, cy, cz = 0.0, 0.0, 0.0
    scale = 1.0  # Default scale
    
    if results.pose_landmarks:
        l_sh = results.pose_landmarks.landmark[11] 
        r_sh = results.pose_landmarks.landmark[12]
        
        # Find the center point between the shoulders
        cx, cy, cz = (l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2, (l_sh.z + r_sh.z)/2
        
        # Calculate the Euclidean distance between the shoulders
        shoulder_dist = np.linalg.norm([l_sh.x - r_sh.x, l_sh.y - r_sh.y, l_sh.z - r_sh.z])
        if shoulder_dist > 0:
            scale = shoulder_dist

    def norm(lm_list, is_face=False):
        if not lm_list: return np.zeros(len(SELECTED_FACE_IDS)*3) if is_face else np.zeros(21*3)
        data = []
        for i, lm in enumerate(lm_list.landmark):
            if is_face and i not in SELECTED_FACE_IDS: continue
            # Apply both translation (- cx) and distance scaling (/ scale)
            data.extend([(lm.x - cx) / scale, (lm.y - cy) / scale, (lm.z - cz) / scale])
        return np.array(data)

    lh, rh = norm(results.left_hand_landmarks), norm(results.right_hand_landmarks)
    face = norm(results.face_landmarks, is_face=True)
    
    if results.pose_landmarks:
        pose = np.array([[(lm.x - cx) / scale, (lm.y - cy) / scale, (lm.z - cz) / scale] 
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*3)
        
    return np.concatenate([pose, face, lh, rh])

SHOULD_DEBUG = True 

print("Processing manually trimmed videos...")
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        vid_folder = os.path.join(VIDEO_PATH, action)
        files = [f for f in os.listdir(vid_folder) if f.endswith(('.mp4', '.avi'))]
        
        for file in files:
            v_path = os.path.join(vid_folder, file)
            save_path = os.path.join(DATA_PATH, action, file.split('.')[0] + '.npy')
            if os.path.exists(save_path): continue

            cap = cv2.VideoCapture(v_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Uniform Sampling: Squeezes/Stretches video to exactly 30 frames
            targets = np.linspace(0, total_f - 1, SEQUENCE_LENGTH, dtype=int) if total_f >= SEQUENCE_LENGTH else np.arange(total_f)
            
            frames_extracted, current_f, out = [], 0, None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if current_f in targets:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(img)
                    
                    if SHOULD_DEBUG:
                        if out is None:
                            h, w = frame.shape[:2]
                            out = cv2.VideoWriter(os.path.join(DEBUG_PATH, "master_debug.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
                        
                        debug_img = frame.copy()
                        mp_drawing.draw_landmarks(debug_img, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        mp_drawing.draw_landmarks(debug_img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(debug_img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
                        # Draw selected face points
                        if res.face_landmarks:
                            for idx, lm in enumerate(res.face_landmarks.landmark):
                                if idx in SELECTED_FACE_IDS:
                                    cv2.circle(debug_img, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

                        out.write(debug_img)
                        cv2.imshow('Master Debug - Visualizing First Video', debug_img)
                        cv2.waitKey(1)

                    frames_extracted.append(extract_and_normalize_keypoints(res))
                current_f += 1
            
            cap.release()
            if out: 
                out.release()
                cv2.destroyAllWindows()
                SHOULD_DEBUG = False # Turn off debugging for all future videos
            
            while len(frames_extracted) < SEQUENCE_LENGTH: 
                frames_extracted.append(frames_extracted[-1])
            
            np.save(save_path, frames_extracted[:SEQUENCE_LENGTH])
            print(f"Processed: {action}/{file}")

print("Success! Data extraction complete.")