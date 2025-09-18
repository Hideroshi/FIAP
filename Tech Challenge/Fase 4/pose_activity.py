from collections import deque
from typing import Any

from numpy import floating

from image import (configure_pose_solutions, configure_hands_solutions,
                   configure_face_solutions, cv2_put_text)
from tqdm import tqdm

from text import ensure_file_exists
from video import (cv2_video_capture, break_video_into_indexed_frames,
                   retrieve_video_capture_properties, cv2_video_writer)

import cv2
import mediapipe as mp
import numpy as np

# 2. Parâmetros
WINDOW_SIZE       = 15
MIN_CONSEC        = 3

# Arms
ARM_THRESH        = 0.04
# Hands
HAND_DISP_THRESH  = 0.02
FINGER_THRESH     = 0.005
AREA_THRESH       = 0.0005
FLOW_THRESH       = 0.5
# Mouth
OPEN_THRESH       = 0.02
SMILE_VERT_MIN    = 0.007
SMILE_VERT_MAX    = 0.02
SMILE_HOR_THRESH  = 0.06
MOUTH_TOP_IDX     = 13 
MOUTH_BOTTOM_IDX  = 14
MOUTH_LEFT_IDX    = 61
MOUTH_RIGHT_IDX   = 291
# Fingers
FINGER_PAIRS      = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12,10),
    "ring":   (16,14),
    "pinky":  (20,18)
}
# Aomomaly
ANOMALY_SPEED_THRESH  = 0.1
ANOMALY_MOUTH_THRESH  = 0.08
ANOMALY_HAND_AREA_VAR = 0.02


def compute_finger_lengths(
    hand_lm: Any
) -> dict[str, float]:
    """
    Compute the normalized Euclidean distance between each fingertip and its MCP joint.

    Parameters
    ----------
    hand_lm : mp.solutions.hands.NormalizedLandmarkList
        The list of 21 hand landmarks in normalized coordinates (x,y,z).
        Must include landmarks for tips and MCP joints as defined in FINGER_PAIRS.

    Returns
    -------
    dict[str, float]
        A mapping from finger name (“thumb”, “index”, “middle”, “ring”, “pinky”)
        to the distance between its tip and MCP, in normalized image units (0–1).

    Notes
    -----
    - Uses the global FINGER_PAIRS dict that maps finger names to landmark indices:
        FINGER_PAIRS = {
            "thumb":  (4, 2),
            "index":  (8, 6),
            "middle": (12,10),
            "ring":   (16,14),
            "pinky":  (20,18)
        }
    - Distance is computed as `sqrt((x_tip–x_mcp)^2 + (y_tip–y_mcp)^2)`.
    """
    lengths = {}
    for name, (tip, mcp) in FINGER_PAIRS.items():
        t = hand_lm.landmark[tip]
        m = hand_lm.landmark[mcp]
        lengths[name] = np.hypot(t.x - m.x, t.y - m.y)
    return lengths

def hand_area_variation(
    buf: list[Any]
) -> floating[Any]:
    """
    Compute the standard deviation of convex-hull areas over a sequence of hand landmarks.

    Parameters
    ----------
    buf : list[NormalizedLandmarkList]
        A temporal buffer of hand landmark sets (each normalized to 0–1).

    Returns
    -------
    floating[Any]
        The standard deviation of the convex hull area (in normalized units)
        across the buffer. Higher values indicate more area variation (opening/closing).

    Notes
    -----
    - Converts each landmark list to an (N×2) NumPy array of (x,y).
    - Computes the convex hull area via OpenCV’s `convexHull` + `contourArea`.
    - Returns `np.std(areas)` for the list of hull areas.
    """
    areas = []
    for h in buf:
        pts = np.array([(lm.x, lm.y) for lm in h.landmark], dtype = np.float32)
        hull = cv2.convexHull(pts)
        areas.append(cv2.contourArea(hull))
    return np.std(areas)

def detect_arm_movement(
    buf: list[Any],
    vis_thresh: float = 0.8
) -> bool:
    """
    Determine whether there is significant arm movement in a buffer of pose landmarks.

    Parameters
    ----------
    buf : list[NormalizedLandmarkList]
        A temporal buffer of pose landmark sets (each normalized to 0–1).
    vis_thresh : float, optional
        Minimum landmark visibility threshold to include that joint in speed calculation.

    Returns
    -------
    bool
        True if the average speed across shoulder, elbow, and wrist joints
        exceeds the global ARM_THRESH constant; False otherwise.

    Notes
    -----
    - Monitored landmark indices: [11,12,13,14,15,16] (left/right shoulder, elbow, wrist).
    - Speed per joint is mean frame-to-frame Euclidean distance.
    - Overall speed is the mean of all joint speeds.
    - Returns False if no joint had ≥2 valid points.
    """
    idxs = [11,12,13,14,15,16]
    speeds = []
    for i in idxs:
        pts = [(lm.landmark[i].x, lm.landmark[i].y)
               for lm in buf if lm.landmark[i].visibility >= vis_thresh]
        if len(pts) >= 2:
            arr = np.array(pts)
            speeds.append(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean())
    return (np.mean(speeds) > ARM_THRESH) if speeds else False

def detect_rich_hand_movement(
    buf: list[Any]
) -> bool:
    """
    Detect hand movement by combining global displacement, finger length variation,
    and area variation metrics.

    Parameters
    ----------
    buf : list[NormalizedLandmarkList]
        A temporal buffer of hand landmark sets.

    Returns
    -------
    bool
        True if any of the following exceed their thresholds:
        - Global hand landmark displacement > HAND_DISP_THRESH
        - Mean finger length variation > FINGER_THRESH
        - Convex hull area variation > AREA_THRESH

    Notes
    -----
    - Global displacement: mean pixel distance across all landmarks frame-to-frame.
    - Finger variation: computed by `compute_finger_lengths` and taking mean
      of absolute differences.
    - Area variation: from `hand_area_variation`.
    """
    arr = np.array([[(lm.x, lm.y) for lm in h.landmark] for h in buf])
    disp = np.linalg.norm(np.diff(arr, axis=0), axis=2).mean()
    fl_buf = [compute_finger_lengths(h) for h in buf]
    finger_vars = [np.mean(np.abs(np.diff([d[f] for d in fl_buf]))) for f in FINGER_PAIRS]
    finger_var = np.mean(finger_vars)
    area_var = hand_area_variation(buf)
    return (disp > HAND_DISP_THRESH) or (finger_var > FINGER_THRESH) or (area_var > AREA_THRESH)

def compute_face_bbox(
    face_lms: Any,
    width: int,
    height: int
) -> tuple[int, int, int, int]:
    """
    Compute a pixel-space bounding box from normalized face landmarks.

    Parameters
    ----------
    face_lms : NormalizedLandmarkList
        The set of facial landmarks in normalized (0–1) coordinates.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        The bounding box as (x1, y1, x2, y2) in pixel coordinates,
        where (x1, y1) is the top-left and (x2, y2) is the bottom-right.

    Notes
    -----
    - Finds the min/max of all landmark x/y and scales by frame dimensions.
    - Useful for cropping or drawing a rectangle around the full face region.
    """
    xs = [lm.x for lm in face_lms.landmark]
    ys = [lm.y for lm in face_lms.landmark]
    x1, x2 = int(min(xs)*width), int(max(xs)*width)
    y1, y2 = int(min(ys)*height), int(max(ys)*height)
    return x1, y1, x2, y2

def compute_mouth_state(fl: Any) -> str:
    """
    Determine whether the mouth is open, smiling, or closed based on FaceMesh landmarks.

    Parameters
    ----------
    fl : NormalizedLandmarkList
        A list of facial landmarks returned by MediaPipe FaceMesh for one face.

    Returns
    -------
    str
        One of {"open", "smile", "closed"} indicating the mouth state.
    """
    top   = fl.landmark[MOUTH_TOP_IDX]
    bot   = fl.landmark[MOUTH_BOTTOM_IDX]
    left  = fl.landmark[MOUTH_LEFT_IDX]
    right = fl.landmark[MOUTH_RIGHT_IDX]

    vert_dist = abs(top.y - bot.y)
    hor_dist  = abs(left.x - right.x)

    xs = [lm.x for lm in fl.landmark]
    face_width = max(xs) - min(xs)
    hor_norm = hor_dist / (face_width + 1e-6)

    if vert_dist > OPEN_THRESH:
        return "open"
    if SMILE_VERT_MIN < vert_dist < SMILE_VERT_MAX and hor_norm > SMILE_HOR_THRESH:
        return "smile"
    return "closed"

def define_mouth_state_and_face_bbox(
    face_mesh: mp.solutions.face_mesh,
    smile_cascade: cv2.CascadeClassifier,
    indexed_frame: cv2.typing.MatLike,
    gray_frame: cv2.typing.MatLike,
    rgb_frame: cv2.typing.MatLike,
    width: int,
    height: int
) -> tuple[str, tuple[int, int, int, int], float]:
    """
    Detect face landmarks, draw face bbox, and classify mouth state (with optional cascade filter).

    Parameters
    ----------
    face_mesh : FaceMesh
        Initialized MediaPipe FaceMesh object.
    smile_cascade : CascadeClassifier
        OpenCV Haar cascade for smile verification.
    indexed_frame : MatLike
        BGR frame on which to draw.
    gray_frame : MatLike
        Grayscale version of the frame.
    rgb_frame : MatLike
        RGB version used for FaceMesh processing.
    width, height : int
        Pixel dimensions of the frame.

    Returns
    -------
    mouth_state : str
        One of {"open", "smile", "closed"}.
    face_bbox : (x1,y1,x2,y2) or None
        Pixel coordinates of the face bounding box if detected.
    mouth_vert : float
        Normalized vertical distance between top and bottom lip landmarks
        (in [0,1] relative to face size) :contentReference[oaicite:4]{index=4}.
    """
    face_bbox = None
    mouth_state = "closed"
    mouth_vert = 0.0

    face_lms_all = face_mesh.process(rgb_frame).multi_face_landmarks
    if face_lms_all:
        fb = face_lms_all[0]
        x1, y1, x2, y2 = compute_face_bbox(fb, width,height)
        face_bbox = (x1, y1, x2, y2)
        top = fb.landmark[MOUTH_TOP_IDX]
        bot = fb.landmark[MOUTH_BOTTOM_IDX]
        mouth_vert = abs(top.y - bot.y)
        mouth_state = compute_mouth_state(fb)
        if mouth_state == "smile":
            mouth_roi = gray_frame[y1:y2, x1:x2]
            if len(smile_cascade.detectMultiScale(
                    mouth_roi, 1.7, 20, minSize=(25, 25))) == 0:
                mouth_state = "closed"
        cv2.rectangle(indexed_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return mouth_state, face_bbox, mouth_vert


def define_rl_regions_of_interest_and_face_touch(
    frame:  cv2.typing.MatLike,
    rgb_frame: cv2.typing.MatLike,
    gray_frame: cv2.typing.MatLike,
    mp_hands: mp.solutions.hands,
    mp_drawing: mp.solutions.drawing_utils,
    hands: mp.solutions.hands.Hands,
    right_buf: list[Any],
    left_buf: list[Any],
    face_bbox: tuple[int, int, int, int],
    width: int,
    height: int
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike, bool, list[float]]:
    """
    Detect hand landmarks, draw them, compute hand ROIs, and check for face touches.

    Parameters
    ----------
    frame : MatLike
        BGR frame for drawing.
    rgb_frame, gray_frame : MatLike
        Converted frames for landmark detection.
    mp_hands, mp_drawing : modules
        MediaPipe Hands utilities for drawing.
    hands : Hands
        Initialized MediaPipe Hands object.
    right_buf, left_buf : lists
        Buffers of past hand landmark lists.
    face_bbox : (x1,y1,x2,y2) or None
        Coordinates of face bbox to check touch.
    width, height : int
        Frame dimensions.

    Returns
    -------
    right_roi, left_roi : MatLike or None
        Grayscale ROIs around each hand for optical flow.
    face_touch : bool
        True if any hand landmark falls inside the face bbox.
    hands_variations:
    """
    face_touch = False
    res_h = hands.process(rgb_frame)
    left_roi = right_roi = None
    if res_h.multi_hand_landmarks:
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            for hlm in res_h.multi_hand_landmarks:
                for lm in hlm.landmark:
                    px, py = int(lm.x * width), int(lm.y * height)
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        face_touch = True
                        break
                if face_touch: break

        for hlm, hd in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            side = hd.classification[0].label
            mp_drawing.draw_landmarks(frame, hlm,
                mp_hands.HAND_CONNECTIONS)
            xs = [lm.x for lm in hlm.landmark];
            ys = [lm.y for lm in hlm.landmark]
            xx1, xx2 = int(min(xs) * width), int(max(xs) * width)
            yy1, yy2 = int(min(ys) * height), int(max(ys) * height)
            pad = 20
            xx1, yy1 = max(0, xx1 - pad), max(0, yy1 - pad)
            xx2, yy2 = min(width, xx2 + pad), min(height, yy2 + pad)
            roi = gray_frame[yy1:yy2, xx1:xx2]

            if side == "Left":
                right_buf.append(hlm)
                right_roi = roi
            else:
                left_buf.append(hlm)
                left_roi = roi

    hands_variations = []
    if len(left_buf)==WINDOW_SIZE:  hands_variations.append(hand_area_variation(left_buf))
    if len(right_buf)==WINDOW_SIZE: hands_variations.append(hand_area_variation(right_buf))
    return right_roi, left_roi, face_touch, hands_variations


def define_prev_lr_roi_and_flow(
    fb_params: dict,
    prev_left_roi: cv2.typing.MatLike,
    prev_right_roi: cv2.typing.MatLike,
    left_roi: cv2.typing.MatLike,
    right_roi: cv2.typing.MatLike
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike, bool, bool]:
    """
    Compute Farneback optical flow between previous and current hand ROIs.

    Parameters
    ----------
    fb_params : dict
        Farneback parameters for calcOpticalFlowFarneback.
    prev_left_roi, prev_right_roi : MatLike or None
        Previous grayscale ROIs for left and right hands.
    left_roi, right_roi : MatLike or None
        Current grayscale ROIs.

    Returns
    -------
    prev_left_roi, prev_right_roi : MatLike or None
        Updated previous ROIs for next iteration.
    left_flow, right_flow : bool
        True if mean flow magnitude exceeds FLOW_THRESH.
    """
    left_flow = right_flow = False
    if prev_left_roi is not None and left_roi is not None and left_roi.size>0:
        rsz = cv2.resize(left_roi, (prev_left_roi.shape[1], prev_left_roi.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(prev_left_roi, rsz, None, **fb_params)
        mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
        left_flow = mag.mean() > FLOW_THRESH
        prev_left_roi = rsz
    elif left_roi is not None and left_roi.size>0:
        prev_left_roi = left_roi

    if prev_right_roi is not None and right_roi is not None and right_roi.size>0:
        rsz = cv2.resize(right_roi, (prev_right_roi.shape[1], prev_right_roi.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(prev_right_roi, rsz, None, **fb_params)
        mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
        right_flow = mag.mean() > FLOW_THRESH
        prev_right_roi = rsz
    elif right_roi is not None and right_roi.size>0:
        prev_right_roi = right_roi
    return prev_left_roi, prev_right_roi, left_flow, right_flow


def detect_arm_and_hands_movements(
    pose_buf: list[Any],
    left_buf: list[Any],
    right_buf: list[Any],
    left_flow: bool,
    right_flow: bool,
    ws: int = WINDOW_SIZE
) -> tuple[bool, bool, bool]:
    """
    Determine whether arm, left hand, or right hand movements are occurring.

    Parameters
    ----------
    pose_buf, left_buf, right_buf : Lists of landmark buffers
        Temporal buffers of landmarks (length up to WINDOW_SIZE).
    left_flow, right_flow : bool
        Optical flow flags for subtle hand movement.
    ws : int
        Required buffer length (window size) for classification.

    Returns
    -------
    arm_move, left_move, right_move : bool
        Flags indicating detected movements for arm, left hand, and right hand.
    """
    arm_move   = (len(pose_buf)==ws and detect_arm_movement(pose_buf))
    left_move  = (len(left_buf)==ws and detect_rich_hand_movement(left_buf)) or left_flow
    right_move = (len(right_buf)==ws and detect_rich_hand_movement(right_buf)) or right_flow
    return arm_move, left_move, right_move


def calculate_and_count_consecutive_movement(
    consec: int,
    move: bool,
    count: int,
    mc: int = MIN_CONSEC
) -> tuple[int, int]:
    """
    Debounce a movement flag: increment consec if move True, else reset, and count event at threshold.

    Parameters
    ----------
    consec : int
        Current consecutive count of frames where move was True.
    move : bool
        Current frame movement flag.
    count : int
        Total event counter.
    mc : int
        Minimum consecutive frames required to register an event.

    Returns
    -------
    new_consec, new_count : int, int
        Updated consecutive count and total event count.
    """
    consec = consec + 1 if move else 0
    if consec == mc: count += 1
    return consec, count


def cv2_put_text_count(
    frame: cv2.typing.MatLike,
    arm_count: int,
    left_count: int,
    right_count: int,
    face_touch_count: int,
    open_count: int,
    close_count: int,
    smile_count: int
) -> None:
    """
    Overlay all movement and expression counts on the frame as text.

    Parameters
    ----------
    frame : MatLike
        BGR image on which to draw.
    *_count : int
        Counters for each event type.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2_put_text(frame, f"Arms: {arm_count}", (20, 80),
                 font, 1, (0, 255, 255))
    cv2_put_text(frame, f"Left Hand: {left_count}", (20, 120),
                 font, 1, (255, 255, 0))
    cv2_put_text(frame, f"Right Hand: {right_count}", (20, 160),
                 font, 1, (255, 0, 255))
    cv2_put_text(frame, f"Face Touch: {face_touch_count}", (20, 200),
                 font, 1, (255, 255, 255))
    cv2_put_text(frame, f"Opened Mouth: {open_count}", (20, 240),
                 font, 1, (0, 128, 255))
    cv2_put_text(frame, f"Closed Mouth: {close_count}", (20, 280),
                 font, 1, (128, 128, 128))
    cv2_put_text(frame, f"Smile: {smile_count}", (20, 320),
                 font, 1, (0, 255, 128))

def fmt(times: list[float]) -> str:
    """
    Format a list of frames into a human-readable preview string.

    Parameters
    ----------
    times : list[float]
        List of event frames.

    Returns
    -------
    str
        A comma-separated preview of up to 5 times (e.g., "1f, 2f, 4f…").
    """
    return ", ".join(f"{t:.1f}f" for t in times[:5]) + ("…" if len(times)>5 else "")


def is_anomalous(
    arm_speeds: list[float],
    hand_area_vars: list[float],
    mouth_vert_dist: float
) -> bool:
    """
    Determine whether motion or facial metrics exceed anomaly thresholds.

    This function evaluates three types of measurements—arm joint speeds,
    hand area variation, and mouth opening—to decide if any behavior
    qualifies as anomalous based on predefined thresholds.

    Parameters
    ----------
    arm_speeds : list[float]
        Mean per-frame speeds for each shoulder, elbow, and wrist joint
        over a rolling window. If non-empty and the maximum speed
        exceeds `ANOMALY_SPEED_THRESH`, the function flags an anomaly.
    hand_area_vars : list[float]
        Standard deviations of the convex-hull area of detected hands
        over a rolling window. A maximum value above `ANOMALY_HAND_AREA_VAR`
        also signals an anomaly.
    mouth_vert_dist : float
        Normalized vertical distance between top and bottom lip landmarks.
        A value greater than `ANOMALY_MOUTH_THRESH` indicates an anomalous
        mouth opening.

    Returns
    -------
    bool
        `True` if **any** of the provided metrics crosses its anomaly
        threshold; otherwise `False`.
    """
    if arm_speeds and max(arm_speeds) > ANOMALY_SPEED_THRESH:
        return True
    if hand_area_vars and max(hand_area_vars) > ANOMALY_HAND_AREA_VAR:
        return True
    if mouth_vert_dist > ANOMALY_MOUTH_THRESH:
        return True
    return False


def compute_arm_speeds(
    buf: list[Any],
    vis_thresh: float = 0.8
) -> list[float]:
    """
    Calculate the average frame-to-frame speed for each shoulder, elbow, and wrist joint.

    Parameters
    ----------
    buf : List[NormalizedLandmarkList]
        A time-ordered buffer of MediaPipe Pose landmark lists (length ≥ 2).
    vis_thresh : float, optional
        Minimum visibility required for a landmark to be included (0–1). Default is 0.8.

    Returns
    -------
    list[float]
        A list of mean speeds (in normalized landmark units per frame) for each joint index
        in [11, 12, 13, 14, 15, 16] (left/right shoulder, elbow, wrist). Joints with fewer
        than 2 valid samples are omitted.
    """
    joint_idxs = [11, 12, 13, 14, 15, 16]
    speeds = []

    for idx in joint_idxs:
        # Gather normalized (x,y) positions for this joint across frames
        pts = [
            (lm.landmark[idx].x, lm.landmark[idx].y)
            for lm in buf
            if lm.landmark[idx].visibility >= vis_thresh
        ]

        # Need at least two points to compute a speed
        if len(pts) < 2:
            continue

        arr = np.array(pts)
        # np.diff computes consecutive displacements :contentReference[oaicite:0]{index=0}
        deltas = np.diff(arr, axis=0)
        # Euclidean norm of each delta vector :contentReference[oaicite:1]{index=1}
        dists = np.linalg.norm(deltas, axis=1)
        speeds.append(float(dists.mean()))

    return speeds


def hand_area_variation(buf: list[Any]) -> float:
    """
    Calculate the variability in hand shape area over time by computing the standard
    deviation of convex hull areas from a sequence of hand landmark sets.
    :contentReference[oaicite:0]{index=0}

    Parameters
    ----------
    buf : list[Any]
        A list of hand landmark objects (e.g., MediaPipe `NormalizedLandmarkList`), each
        containing normalized (x, y) coordinates for all detected hand landmarks.
        :contentReference[oaicite:1]{index=1}

    Returns
    -------
    float
        The standard deviation of the convex hull areas (in normalized units) calculated
        across all frames in `buf`. A higher value indicates greater variation in hand
        shape (e.g., opening/closing the palm). Returns 0.0 if `buf` has fewer than two entries. :contentReference[oaicite:2]{index=2}

    Notes
    -----
    - Uses `cv2.convexHull(...)` to compute the convex hull of each set of landmarks,
      then measures its area with `cv2.contourArea(...)`. :contentReference[oaicite:3]{index=3}
    - Applies `numpy.std(...)` to the list of hull areas to quantify dispersion. :contentReference[oaicite:4]{index=4}
    """
    areas = []
    for h in buf:
        pts = np.array([(lm.x, lm.y) for lm in h.landmark], dtype=np.float32)
        hull = cv2.convexHull(pts)
        areas.append(float(cv2.contourArea(hull)))
    return float(np.std(areas))


def detect_pose_and_activities(video_in_path: str, video_out_path: str) -> None:
    """
    Process a video to detect and annotate:
      - Arm movements
      - Left- and right-hand movements
      - Hand touches to the face
      - Mouth states (open, smile, closed)

    The function writes an annotated output video and prints/saves
    a summary of counts and example frame indices for each event.

    Pipeline steps
    --------------
    1. Open and index all frames from the input video.
    2. Initialize MediaPipe Pose, Hands, and FaceMesh solutions,
       plus an OpenCV Haar cascade for smile filtering.
    3. Maintain rolling buffers of landmarks (window size = WINDOW_SIZE)
       for temporal classification of movements.
    4. For each frame:
       a. Convert to grayscale and RGB for processing.
       b. Detect face + mouth state via FaceMesh, with smile verification.
       c. Detect pose landmarks and accumulate in `pose_buf`.
       d. Detect hand landmarks, build per-hand ROIs, check "face touch".
       e. Compute optical flow between consecutive ROIs to catch subtle hand motion.
       f. Classify arm/hand movement when buffer is full.
       g. Debounce each event: require MIN_CONSEC consecutive frames before counting.
       h. Append frame index to per-event timestamp lists for summary.
       i. Draw all counts on the frame and write/display it.
    5. Release resources and close windows.
    6. Build and print/save a text summary listing:
       - Total counts of each movement/emotion.
       - Example frame indices where they occurred.

    Parameters
    ----------
    video_in_path : str
        Path to the input video file.
    video_out_path : str
        Path where the annotated output video will be saved.

    Returns
    -------
    None
        Outputs an annotated video file and a summary text file at:
        "./doc/videos/result/tc4_pose_activity_summary.txt".
    """
    video_capture  = cv2_video_capture(video_in_path)
    indexed_frames = break_video_into_indexed_frames(video_capture)
    video_writer   = cv2_video_writer(video_capture, video_out_path)
    width, height, _, total_frames, _ = retrieve_video_capture_properties(video_capture)

    mp_pose, pose = configure_pose_solutions()
    mp_hands, hands = configure_hands_solutions()
    _, face_mesh = configure_face_solutions()
    mp_drawing = mp.solutions.drawing_utils
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    pose_buf  = deque(maxlen=WINDOW_SIZE)
    left_buf  = deque(maxlen=WINDOW_SIZE)
    right_buf = deque(maxlen=WINDOW_SIZE)

    arm_count = left_count = right_count = face_touch_count = \
        open_count = smile_count = close_count = \
        arm_consec = left_consec = right_consec = face_consec = \
        open_consec = smile_consec = close_consec = 0

    arm_times = []
    left_times = []
    right_times = []
    face_times = []
    open_times = []
    smile_times = []
    close_times = []
    anomaly_times = []

    prev_left_roi = prev_right_roi = None
    fb_params = dict(pyr_scale = 0.5, levels = 3, winsize = 15,
                     iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)

    for idx, indexed_frame in tqdm(indexed_frames, desc="Analysing video frames"):
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

        gray_frame = cv2.cvtColor(indexed_frame, cv2.COLOR_BGR2GRAY)
        rgb_frame  = cv2.cvtColor(indexed_frame, cv2.COLOR_BGR2RGB)

        mouth_state, face_bbox, mouth_vert = define_mouth_state_and_face_bbox(face_mesh,
            smile_cascade, indexed_frame, gray_frame, rgb_frame, width, height)

        plm = pose.process(rgb_frame).pose_landmarks
        arm_speeds = []
        if plm:
            mp_drawing.draw_landmarks(indexed_frame, plm, mp_pose.POSE_CONNECTIONS)
            pose_buf.append(plm)
            arm_speeds = compute_arm_speeds(pose_buf)

        right_roi, left_roi, face_touch, hand_area_vars = (
            define_rl_regions_of_interest_and_face_touch(
                indexed_frame, rgb_frame, gray_frame, mp_hands, mp_drawing,
                hands, right_buf, left_buf, face_bbox, width, height))

        prev_left_roi, prev_right_roi, left_flow, right_flow = (
            define_prev_lr_roi_and_flow(
                fb_params, prev_left_roi, prev_right_roi, left_roi, right_roi))

        arm_move, left_move, right_move = (
            detect_arm_and_hands_movements(
                pose_buf, left_buf, right_buf, left_flow, right_flow))

        arm_consec, arm_count = calculate_and_count_consecutive_movement(
            arm_consec, bool(arm_speeds and np.mean(arm_speeds) > ARM_THRESH), arm_count)
        left_consec, left_count = calculate_and_count_consecutive_movement(
            left_consec, left_move, left_count)
        right_consec, right_count = calculate_and_count_consecutive_movement(
            right_consec, right_move, right_count)
        face_consec, face_touch_count = calculate_and_count_consecutive_movement(
            face_consec, face_touch, face_touch_count)
        if mouth_state == "smile" and open_consec > 0:
            smile_consec, smile_count = calculate_and_count_consecutive_movement(
                smile_consec, True, smile_count)
            open_consec, open_count = calculate_and_count_consecutive_movement(
                open_consec, True, open_count)
        elif mouth_state == "smile" and close_consec > 0:
            smile_consec, smile_count = calculate_and_count_consecutive_movement(
                smile_consec, True, smile_count)
            close_consec, close_count = calculate_and_count_consecutive_movement(
                close_consec, True, close_count)
        if mouth_state == "open":
            open_consec, open_count = calculate_and_count_consecutive_movement(
                open_consec, mouth_state == "open", open_count)
            smile_consec, smile_count = calculate_and_count_consecutive_movement(
                smile_consec, False, smile_count)
            close_consec, close_count = calculate_and_count_consecutive_movement(
                close_consec, False, close_count)
        if mouth_state == "closed":
            close_consec, close_count = calculate_and_count_consecutive_movement(
                close_consec, mouth_state == "closed", close_count)
            smile_consec, smile_count = calculate_and_count_consecutive_movement(
                smile_consec, False, smile_count)
            open_consec, open_count = calculate_and_count_consecutive_movement(
                open_consec, False, open_count)

        arm_times.append(idx)
        left_times.append(idx)
        right_times.append(idx)
        face_times.append(idx)
        open_times.append(idx)
        smile_times.append(idx)
        close_times.append(idx)
        if is_anomalous(arm_speeds, hand_area_vars, mouth_vert):
            anomaly_times.append(idx)
            cv2_put_text(indexed_frame, "ANOMALY!", (width - 200, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 3)

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            text = f"Mouth: {mouth_state}"
            text_x = x1 + 10
            text_y = y2 - 10
            cv2_put_text(indexed_frame, text, (text_x, text_y),
             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        cv2_put_text_count(indexed_frame, arm_count, left_count, right_count,
                             face_touch_count, open_count, close_count, smile_count)
        video_writer.write(indexed_frame)
        cv2.imshow("Analyzed Video", indexed_frame)

    video_capture.release()
    video_writer.release()
    pose.close()
    hands.close()
    face_mesh.close()
    cv2.destroyAllWindows()

    summary = f"""
    ========================= VIDEO SUMMARY: ACTIVITIES =========================
    Total frames analyzed:    {total_frames}
    Total anomalies detected: {len(anomaly_times)}  (ex.: {fmt(anomaly_times)})

    Counts of detected events:
    - Arm movements:         {arm_count} times (e.g.: {fmt(arm_times)})
    - Left hand movements:   {left_count} times (e.g.: {fmt(left_times)})
    - Right hand movements:  {right_count} times (e.g.: {fmt(right_times)})
    - Hand touches to face:  {face_touch_count} times (e.g.: {fmt(face_times)})

    Facial expressions:
    - Mouth open:            {open_count} times (e.g.: {fmt(open_times)})
    - Smile:                 {smile_count} times (e.g.: {fmt(smile_times)})
    - Mouth closed:          {close_count} times (e.g.: {fmt(close_times)})
    """
    write_summary_analysis(summary = summary)


def write_summary_analysis(
    analysis_output_path: str = "./doc/videos/result/summary_analysis.txt",
    summary: str = ""
) -> None:
    """
    Append a summary of emotion-appearance analysis to a text file.

    Parameters
    ----------
    analysis_output_path : str
        Filesystem path to the summary file. If the file does not already exist,
        it will be created (including any missing parent directories).

    summary : str
        Content to be written to a summary file.

    Returns
    -------
    None
        This function does not return a value; its effect is writing to disk.
    """
    ensure_file_exists(analysis_output_path)
    with open(analysis_output_path, "a", encoding="utf-8") as f:
        print(summary, file=f)


if __name__ == "__main__":

    video_in_path = "./doc/videos/tc4_video.mp4"
    video_out_path = "./doc/videos/result/tc4_video_pa.mp4"
    detect_pose_and_activities(video_in_path, video_out_path)
