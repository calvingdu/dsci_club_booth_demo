import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
from collections import defaultdict

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hdist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hdist)

def mouth_open_ratio(landmarks, w, h):
    up = np.array([landmarks[13].x * w, landmarks[13].y * h])
    low = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right = np.array([landmarks[291].x * w, landmarks[291].y * h])
    mouth_open = np.linalg.norm(up - low)
    mouth_width = np.linalg.norm(left - right)
    return mouth_open / mouth_width if mouth_width > 0 else 0

def smile_intensity(landmarks, w, h):
    left = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right = np.array([landmarks[291].x * w, landmarks[291].y * h])
    top = np.array([landmarks[13].x * w, landmarks[13].y * h])
    bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])

    mouth_width = np.linalg.norm(left - right)
    mouth_height = np.linalg.norm(top - bottom)

    intensity = mouth_height / mouth_width if mouth_width > 0 else 0
    intensity = max(0, min(1, intensity * 2.0))

    return intensity

def head_tilt(landmarks, w, h):
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    dx, dy = chin - nose
    angle = math.degrees(math.atan2(dx, dy))  # tilt in degrees
    return angle

# faculty list
FACULTIES = [
    "Arts", "Science", "Engineering", "Sauder", "Kinesiology"
]

def faculty_scores(tiredness, smile, tilt, mouth_ratio):
    # normalize tilt into [-1, 1]
    tilt_score = max(-1, min(1, tilt / 30.0))  

    scores = {
        # Engineering: high tiredness, low smile
        "Engineering": (tiredness / 100) * 1.2,

        # Arts: mostly driven by smile intensity
        "Arts": smile * 1.3,

        # Kinesiology: needs both a smile and upright posture
        "Kinesiology": smile * (1 - abs(tilt_score)) * 1.3,

        # Sauder: tilted head (confidence) and smiling
        "Sauder": max(0, tilt_score)*1.4 + (smile) * 0.8,

        # Science: fallback when posture is off and smile is low
        "Science": (1 - smile) * (1 - abs(tilt_score)) * 1.1 + 0.2
    }
    return scores


def predict_faculty(tiredness, smile, tilt, mouth_ratio):
    scores = faculty_scores(tiredness, smile, tilt, mouth_ratio)
    chosen = max(scores, key=scores.get)
    return chosen, scores


# eye indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# webcam setup
cap = cv2.VideoCapture(0)
mode = 'default'
blink_count = 0
blink_frame = False
last_seen_time = time.time()
reset_timeout = 1.0  # reset if no face for 1s

faculty_fixed = None
faculty_scores_fixed = None
frozen_metrics = defaultdict(float)
faculty_lock_time = 0.5
decision_delay = 2.0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            last_seen_time = time.time()  # reset timeout
            for lm in results.multi_face_landmarks:
                # eye aspect ratio
                ear_left = eye_aspect_ratio(lm.landmark, LEFT_EYE, w, h)
                ear_right = eye_aspect_ratio(lm.landmark, RIGHT_EYE, w, h)
                ear = (ear_left + ear_right) / 2.0

                mouth_ratio = mouth_open_ratio(lm.landmark, w, h)
                smile = smile_intensity(lm.landmark, w, h)
                tilt = head_tilt(lm.landmark, w, h)

                tiredness = (1 - ear) * 100   # higher if eyes are closed
                tiredness = min(100, max(0, int(tiredness)))


                # handle faculty locking
                if faculty_fixed is None:
                    if faculty_lock_time is None:
                        faculty_lock_time = time.time()  # start timer
                    elif time.time() - faculty_lock_time > decision_delay:
                        faculty_fixed, faculty_scores_fixed = predict_faculty(
                            tiredness, smile, tilt, mouth_ratio
                        )
                        frozen_metrics = {
                        "smile": smile,
                        "tilt": tilt,
                        "mouth_ratio": mouth_ratio,
                        "tiredness": tiredness
                        }
        else:
            if time.time() - last_seen_time > reset_timeout:
                blink_count = 0
                tiredness = 0
                faculty_fixed = None
                faculty_scores_fixed = None
                faculty_lock_time = None

        # drawing section
        x, y = 30, 60
        line_spacing = 35
        stats_y_start = y + 40  
        if faculty_fixed:
            text = f"Predicted Faculty: {faculty_fixed}"
            (tw, th), _ = cv2.getTextSize(text,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, 2)

            stats_height = 250
            stats_width = max(tw + 20, 380)
            cv2.rectangle(frame, (x-10, y-th-10),
                        (x-10+stats_width, y-th-10+stats_height),
                        (0, 0, 0), -1)

            # predict
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)


            # demo reasoning
            top_fac = max(faculty_scores_fixed, key=faculty_scores_fixed.get)
            reason = ""
            if top_fac == "Engineering":
                reason = f"sad and tired"
            elif top_fac == "Arts":
                reason = f"so happy"
            elif top_fac == "Sauder":
                reason = f"confident"
            elif top_fac == "Kinesiology":
                reason = f"happy + posture"
            elif top_fac == "Science":
                reason = f"lowkey sad"

            cv2.putText(frame, f"Reason: {reason}",
                        (x, stats_y_start + 4*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
        else:
            cv2.putText(frame, "Thinking... hold still",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (200, 200, 200), 2)

        if mode == 'default':
            cv2.putText(frame, f"Smile Intensity: {frozen_metrics['smile']:.2f}", (x, stats_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Head Tilt: {frozen_metrics['tilt']:.1f} deg", (x, stats_y_start + line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
            cv2.putText(frame, f"Yawn Ratio: {frozen_metrics['mouth_ratio']*100:.1f}%", (x, stats_y_start + 2*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            cv2.putText(frame, f"Tiredness Score: {frozen_metrics['tiredness']}", (x, stats_y_start + 3*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
        elif mode == 'facemesh':
            cv2.putText(frame, f"Smile Intensity: {smile:.2f}", (x, stats_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Head Tilt: {tilt:.1f} deg", (x, stats_y_start + line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
            cv2.putText(frame, f"Yawn Ratio: {mouth_ratio*100:.1f}%", (x, stats_y_start + 2*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            cv2.putText(frame, f"Tiredness Score: {tiredness}", (x, stats_y_start + 3*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            mp_drawing.draw_landmarks(
                frame, lm, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))
            

            # live faculty scores
            y_offset = stats_y_start + 4*line_spacing + 50
            faculty_rectangle = (x-10, y_offset-30, 360, 210)
            cv2.rectangle(frame, (faculty_rectangle[0], faculty_rectangle[1]),
                        (faculty_rectangle[0]+faculty_rectangle[2], faculty_rectangle[1]+faculty_rectangle[3]),
                        (80, 80, 80), -1)
            
            cv2.putText(frame, "Live Faculty Scores:", (faculty_rectangle[0]+40, faculty_rectangle[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            
            live_scores = faculty_scores(tiredness, smile, tilt, mouth_ratio)
            y_offset_text = y_offset + 25
            for i, (fac, score) in enumerate(live_scores.items()):
                cv2.putText(frame, f"{fac}: {score:.2f}", (30, y_offset_text + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        # mode instructions
        cv2.rectangle(frame, (5, h-40), (w-5, h-5), (50, 50, 50), -1)
        cv2.putText(frame, f"Mode: {mode.upper()} (press d = default, f = details, r=refresh, ESC=quit)",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200,200,200), 1)

        cv2.imshow("UBC Faculty Predictor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            mode = 'default'
        elif key == ord('f'):
            mode = 'facemesh'
        elif key == ord('r'):  # refresh
            faculty_fixed = None
            faculty_scores_fixed = None
            faculty_lock_time = None
        elif key == 27:  # esc
            break

cap.release()
cv2.destroyAllWindows()
