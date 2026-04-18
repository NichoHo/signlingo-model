import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants and definitions from extraction.py
ACTIONS = np.array(['Apa', 'Apa Kabar', 'Bagaimana', 'Baik', 'Belajar', 'Berapa', 'Berdiri', 'Bingung', 'Dia', 'Dimana', 'Duduk', 'Halo', 'Kalian', 'Kami', 'Kamu', 'Kapan', 'Kemana', 'Kita', 'Makan', 'Mandi', 'Marah', 'Melihat', 'Membaca', 'Menulis', 'Mereka', 'Minum', 'Pendek', 'Ramah', 'Sabar', 'Saya', 'Sedih', 'Selamat Malam', 'Selamat Pagi', 'Selamat Siang', 'Selamat Sore', 'Senang', 'Siapa', 'Terima Kasih', 'Tidur', 'Tinggi'])

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
MODEL_PATH = 'signlingo_v2_gru.h5'

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


def main():
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    sequence = []
    
    # State tracking
    state = "IDLE"
    last_prediction_str = None
    countdown_start_time = 0
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("Starting webcam...")
        print("Press 'SPACE' to record a sign, and 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Please ensure:")
                print("1. Your Mac System Settings -> Privacy & Security -> Camera has granted permissions to your Terminal / IDE.")
                print("2. You might need to restart your Terminal / IDE after granting permission.")
                print("3. No other application is currently using your webcam.")
                break
            
            # Flip the frame for a mirror effect, easier to use that way
            frame = cv2.flip(frame, 1)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make Detections
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks (optional, helps with debugging)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Logic based on state
            if state == "RECORDING":
                keypoints = extract_and_normalize_keypoints(results)
                sequence.append(keypoints)
                
                # Show recording progress
                cv2.rectangle(image, (0,0), (640, 40), (0, 0, 255), -1)  # Red banner
                cv2.putText(image, f'RECORDING: {len(sequence)}/{SEQUENCE_LENGTH} frames', (10,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if len(sequence) == SEQUENCE_LENGTH:
                    # Run prediction
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    prediction_idx = np.argmax(res)
                    prob = res[prediction_idx]
                    
                    last_prediction_str = f"Result: {ACTIONS[prediction_idx]} ({prob*100:.1f}%)"
                    
                    # Print probabilities to terminal
                    print("\n" + "="*40)
                    print("PREDICTION RESULTS (Sorted by Match %)")
                    print("="*40)
                    
                    action_probs = [(ACTIONS[i], res[i] * 100) for i in range(len(ACTIONS))]
                    action_probs.sort(key=lambda x: x[1], reverse=True)
                    
                    for action_name, probability in action_probs:
                        # Print every word cleanly, left-aligned standard width, right-aligned percentage
                        print(f"  {action_name:.<25} {probability:>6.2f}%")
                        
                    print("="*40 + "\n")
                    
                    # Reset back to IDLE
                    state = "IDLE"
                    
            elif state == "IDLE":
                # Display instruction banner
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1) # Orange banner
                cv2.putText(image, "Press 'SPACE' to start recording", (10,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display the last prediction if we have one
                if last_prediction_str:
                    cv2.rectangle(image, (0, 40), (640, 80), (0, 128, 0), -1) # Green banner
                    cv2.putText(image, last_prediction_str, (10,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            elif state == "COUNTDOWN":
                time_elapsed = time.time() - countdown_start_time
                time_left = 5.0 - time_elapsed
                
                if time_left <= 0:
                    state = "RECORDING"
                    sequence = []
                else:
                    # Display countdown
                    cv2.rectangle(image, (0,0), (640, 80), (0, 165, 255), -1)  # Yellow-orange banner
                    cv2.putText(image, f"GET READY: {int(time_left) + 1}", (10,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

                 
            # Show to screen
            cv2.imshow('SignLingo Trigger Tester', image)

            # Key handling
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # spacebar
                if state == "IDLE":
                    state = "COUNTDOWN"
                    countdown_start_time = time.time()
                    last_prediction_str = None
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
