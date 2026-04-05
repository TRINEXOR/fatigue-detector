import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64, cv2, numpy as np
from collections import deque
from scipy.spatial import distance as dist
import mediapipe as mp

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "neuralwatch-secret-2024")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE          = [362, 385, 387, 263, 373, 380]
RIGHT_EYE         = [33,  160, 158, 133, 153, 144]
FACE_POSE_INDICES = [1, 152, 263, 33, 287, 57]

MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# Config 
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 2
PERCLOS_WINDOW    = 30
FATIGUE_PERCLOS   = 0.35
PITCH_THRESH      = 15.0
HEAD_NOD_WINDOW   = 60
HEAD_NOD_COUNT    = 3

# Session state 
sessions = {}

def new_session():
    return {
        "blink_counter": 0,
        "total_blinks":  0,
        "ear_history":   deque(maxlen=PERCLOS_WINDOW),
        "pitch_history": deque(maxlen=HEAD_NOD_WINDOW),
    }

# Helpers 
def eye_aspect_ratio(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def get_pts(lm, indices, w, h):
    return np.array([(lm[i].x * w, lm[i].y * h) for i in indices], dtype=np.float64)

def head_pose(lm, w, h):
    img_pts  = get_pts(lm, FACE_POSE_INDICES, w, h)
    focal    = float(w)
    cam_mat  = np.array([[focal, 0, w/2],[0, focal, h/2],[0, 0, 1]], dtype=np.float64)
    dist_co  = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, img_pts, cam_mat, dist_co,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0
    rmat, _  = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1])

# Routes 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# Socket events 
@socketio.on("connect")
def on_connect(auth=None):
    sessions[request.sid] = new_session()

@socketio.on("disconnect")
def on_disconnect():
    sessions.pop(request.sid, None)

@socketio.on("frame")
def on_frame(data):
    sid   = request.sid
    state = sessions.get(sid)
    if state is None:
        return

    # Decode frame
    try:
        _, encoded = data["image"].split(",", 1)
        arr   = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    ear = 0.0; perclos = 0.0; pitch = 0.0; yaw = 0.0; nod_count = 0
    face_detected = False

    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark

        # Stream 1 — Eye blink
        l_pts = get_pts(lm, LEFT_EYE,  w, h)
        r_pts = get_pts(lm, RIGHT_EYE, w, h)
        ear   = (eye_aspect_ratio(l_pts) + eye_aspect_ratio(r_pts)) / 2.0

        state["ear_history"].append(1 if ear < EAR_THRESHOLD else 0)
        perclos = sum(state["ear_history"]) / max(len(state["ear_history"]), 1)

        if ear < EAR_THRESHOLD:
            state["blink_counter"] += 1
        else:
            if state["blink_counter"] >= EAR_CONSEC_FRAMES:
                state["total_blinks"] += 1
            state["blink_counter"] = 0

        # Stream 2 — Head pose
        pitch, yaw = head_pose(lm, w, h)
        state["pitch_history"].append(pitch)

        if len(state["pitch_history"]) == HEAD_NOD_WINDOW:
            arr_p     = np.array(state["pitch_history"])
            crossings = np.where(np.diff(np.sign(arr_p - PITCH_THRESH)))[0]
            nod_count = len(crossings) // 2

    # Dual-stream fusion
    eye_fatigue  = perclos > FATIGUE_PERCLOS
    head_fatigue = nod_count >= HEAD_NOD_COUNT
    fatigue_level = int(eye_fatigue) + int(head_fatigue)

    emit("result", {
        "face":          face_detected,
        "ear":           round(ear, 3),
        "perclos":       round(perclos * 100, 1),
        "blinks":        state["total_blinks"],
        "pitch":         round(pitch, 1),
        "yaw":           round(yaw, 1),
        "nod_count":     nod_count,
        "eye_fatigue":   eye_fatigue,
        "head_fatigue":  head_fatigue,
        "fatigue_level": fatigue_level,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
