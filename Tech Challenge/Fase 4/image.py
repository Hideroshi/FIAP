from deepface import DeepFace
from tqdm import tqdm

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import typing


def load_image_face_encodings_and_names(
    images_path: str
) -> tuple[list[np.array], list[str]]:
    """
    Load face encodings and corresponding person names from a directory of images.

    This function scans the given directory for image files, downsamples each image
    to speed up processing, detects faces, computes face encodings, and derives a
    label from each filename.

    Parameters
    ----------
    images_path : str
        Path to the directory containing reference images. Each image filename is
        expected to encode the person's name (e.g. "01_Alice.jpg" or "02_Bob.png").

    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        - A list of 128-dimensional face encoding arrays (one per image, if a face
          was found).
        - A list of names corresponding to each encoding, extracted from the
          filename (everything between the first two and last two characters
          before the extension by default).

    Notes
    -----
    - Images are resized by a factor of 0.125 in each dimension to speed up
      face detection/encoding.
    - Uses `load_face_locations` and `load_face_encodings` (with model="large"),
      each assumed to wrap calls to `face_recognition.face_locations` and
      `face_recognition.face_encodings`.
    - Filenames are parsed with `os.path.splitext`; the slice `[2:-2]` assumes your
      naming convention places the name between characters 2 and 2 before the
      extension (e.g. "01_Alice01.jpg" → "Alice").
    """
    image_face_encodings = []
    image_face_names     = []
    filenames             = os.listdir(images_path)
    # Percorrer todos os arquivos na pasta fornecida
    for filename in tqdm(filenames, desc = "Loading face encodings and names"):
        # Load image
        image_file = face_recognition.load_image_file(
            os.path.join(images_path, filename)
        )
        image_file = cv2.resize(image_file, (0, 0), fx = 0.125, fy = 0.125)

        face_locations = load_face_locations(image_file)
        face_encodings = load_face_encodings(
            image_file, face_locations, 40,"large")

        if face_encodings:
            image_face_encodings.append(face_encodings[0])
            image_face_names.append(os.path.splitext(filename)[0][2:-2])

    return image_face_encodings, image_face_names


def bgr_2_rgb_frame(bgr_frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert an OpenCV BGR frame to an RGB frame and downsample it.

    This helper flips the color channels from BGR (used by OpenCV) to RGB
    (expected by libraries like face_recognition), ensures the array is
    contiguous in memory for performance, and then downsamples the image
    by a factor of 8 in each dimension to speed up subsequent face detection.

    Parameters
    ----------
    bgr_frame : MatLike
        Input image in BGR color space (height x width x 3), as returned
        by `cv2.VideoCapture.read()` or similar.

    Returns
    -------
    MatLike
        Output image in RGB color space, downsampled to 12.5% of the
        original width and height. The result is contiguous in memory
        (C‐order) and suitable for libraries that require RGB input.

    Notes
    -----
    - Uses `np.ascontiguousarray` to ensure efficient memory layout.
    - Uses `cv2.resize` with `fx=0.125, fy=0.125` to reduce resolution.
    - Default interpolation (`INTER_LINEAR`) is typically sufficient, but
      you can specify another method if desired.
    """
    rgb_frame =  np.ascontiguousarray(bgr_frame[:, :, ::-1])
    return cv2.resize(
        rgb_frame,
        (0, 0),
        fx = 0.125,
        fy = 0.125
    )


def load_face_locations(
    rgb_frame: cv2.typing.MatLike,
    number_of_times_to_upsample: int = 5,
    #model: str = "hog"
    model: str = "cnn"
) -> list[tuple[int,int,int,int]]:
    """
    Detect the pixel coordinates of all faces in an RGB image.

    This wraps `face_recognition.face_locations`, which returns a list
    of bounding boxes in (top, right, bottom, left) order for each face
    detected in the image.

    Parameters
    ----------
    rgb_frame : MatLike
        The input image in RGB format (height x width x 3). If you have a BGR
        image from OpenCV, convert it first (e.g. `frame[:, :, ::-1]`).
    number_of_times_to_upsample : int, optional
        Number of times to upsample the image before running face detection.
        Higher values can find smaller faces but will be slower. Default is 2.
    model : str, optional
        The underlying detection model to use:
        - "hog": Histogram of Oriented Gradients (fast CPU method).
        - "cnn": Convolutional Neural Network (more accurate, requires GPU).
        Default is "hog".

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of tuples, one per detected face, in the format
        `(top, right, bottom, left)`, representing pixel coordinates
        of the face bounding box.
    """
    return face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample,
        model
    )


def load_face_encodings(
    rgb_frame: cv2.typing.MatLike,
    known_face_locations: None | list[tuple[int,int,int,int]] = None,
    num_jitters:int      = 1,
    model: str           = "small"
) -> list[np.ndarray]:
    """
    Compute face encodings for one or more faces in an RGB image.

    This wraps `face_recognition.face_encodings`, returning a list of 128-dimensional
    encoding vectors (one per detected face) which can be used for face comparison.

    Parameters
    ----------
    rgb_frame : MatLike
        The input image in RGB color space (height x width x 3). If using an OpenCV
        BGR image, convert it first (`rgb_frame = bgr_frame[:, :, ::-1]`).
    known_face_locations : list[tuple[int, int, int, int]], default=None
        An optional list of face bounding boxes in `(top, right, bottom, left)` order.
        If provided, only these regions will be encoded. If None, faces are located
        automatically using the HOG model.
    num_jitters : int, default=1
        How many times to re-sample each face when calculating encodings. Higher values
        increase robustness at the cost of speed (e.g. 100x slower for num_jitters=100).
    model : str, default="small"
        Which underlying neural network model to use:
        - `"small"`: faster, returns 5-point encodings.
        - `"large"`: more accurate 128-point encodings, but slower.

    Returns
    -------
    list[np.ndarray]
        A list of face encoding arrays (each of shape (128,)) corresponding to each face
        found or provided in the image.

    Raises
    ------
    ValueError
        If `model` is not `"small"` or `"large"`.
    """
    if model not in ("small", "large"):
        raise ValueError(f"model must be 'small' or 'large', not '{model}'")

    return face_recognition.face_encodings(
        face_image           = rgb_frame,
        known_face_locations = known_face_locations,
        num_jitters          = num_jitters,
        model                = model
    )


def load_face_locations_and_encodings(
    rgb_frame: cv2.typing.MatLike,
    number_of_times_to_upsample: int = 2,
    num_jitters: int                 = 2,
    model: str                       = "large"
)-> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """
    Detects faces in an RGB image and computes their encodings in one call.

    This function first locates every face in the image using `load_face_locations`,
    then passes those bounding boxes into `load_face_encodings` to get a 128-dimensional
    encoding vector for each detected face.

    Parameters
    ----------
    rgb_frame : MatLike
        The input image in RGB color space (height x width x 3). If your image is in
        BGR (from OpenCV), convert it first via `rgb_frame = bgr_frame[:, :, ::-1]`.
    number_of_times_to_upsample : int, default=2
        How many times to upsample the image when searching for faces. Higher values
        can detect smaller faces but are slower.
    num_jitters : int, default=2
        How many times to re-sample each face when computing encodings. More jitters
        can improve accuracy at the cost of performance.
    model : str, default="large"
        Which neural network model to use for encoding:
        - `"small"`: faster, lower-dimensional (5-point) encodings.
        - `"large"`: slower, full (128-point) encodings.

    Returns
    -------
    tuple[list[tuple[int, int, int, int]], list[np.ndarray]]
        - A list of face bounding boxes, each as a `(top, right, bottom, left)` tuple.
        - A list of NumPy arrays, each a 128-dimensional face encoding corresponding
          to the detected faces in the same order.
    """
    face_locations = load_face_locations(
        rgb_frame,
        number_of_times_to_upsample = number_of_times_to_upsample
    )
    return face_locations, load_face_encodings(
        rgb_frame,
        face_locations,
        num_jitters = num_jitters,
        model       = model
    )


def find_name(
    known_face_encodings: list[np.ndarray],
    face_encoding_to_check: np.ndarray,
    known_face_names: list[str],
    tolerance: float = 0.6
) -> str:
    """
    Compare a single face encoding against a list of known encodings and return the best match name.

    This function uses two steps:
      1. A boolean match test with `compare_faces` to see which known encodings are within
         the specified distance (`tolerance`) of the target encoding.
      2. A distance-based selection with `face_distance` to find the closest known encoding.

    Parameters
    ----------
    known_face_encodings : list[np.ndarray]
        A list of 128-dimensional face encoding arrays for known identities.
    face_encoding_to_check : np.ndarray
        A single 128-dimensional face encoding to compare against the known list.
    known_face_names : list[str]
        A list of names corresponding to each encoding in `known_face_encodings`.
    tolerance : float, optional
        Maximum allowed face distance for a positive match. Lower values are more strict.
        Default is 0.6, which is typical for good accuracy on frontal faces.

    Returns
    -------
    str
        The name of the best-matching known face if its distance is within `tolerance`;
        otherwise returns `"Unknown"`.
    """
    compare_faces = face_recognition.compare_faces(
        known_face_encodings   = known_face_encodings,
        face_encoding_to_check = face_encoding_to_check,
        tolerance              = tolerance
    )
    best_match_index = np.argmin(
        face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    )
    if compare_faces[best_match_index]:
        return known_face_names[best_match_index]
    return "Unknown"


def cv2_rectangle(
    frame: cv2.typing.MatLike,
    point1: cv2.typing.Point,
    point2: cv2.typing.Point,
    color: cv2.typing.Scalar = (255, 0, 0), # (B,G,R)
    thickness: int = 4
) -> None:
    """
    Draw a rectangle on an image using OpenCV.

    This is a thin wrapper around `cv2.rectangle` that draws directly
    onto the provided frame (in place).

    Parameters
    ----------
    frame : MatLike
        The image (height × width × channels) on which to draw. This array
        will be modified in place.
    point1 : Point
        The top-left corner of the rectangle, as an (x, y) tuple or
        cv2 Point.
    point2 : Point
        The bottom-right corner of the rectangle, as an (x, y) tuple or
        cv2 Point.
    color : Scalar, optional
        Rectangle color in BGR order. Default is (255, 0, 0) (blue).
    thickness : int, optional
        Line thickness in pixels. If negative, the rectangle will be filled.
        Default is 4.

    Returns
    -------
    None
        The function draws on `frame` and does not return a new image.
    """
    cv2.rectangle(frame, point1, point2, color, thickness)


def cv2_put_text(
    frame: cv2.typing.MatLike,
    text: str,
    org: cv2.typing.Point,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.9,
    color: cv2.typing.Scalar = (0, 255, 0), # (B,G,R)
    thickness: int = 2
) -> None:
    """
    Draw text on an image frame using OpenCV.

    This function wraps `cv2.putText`, drawing the specified string onto
    the provided frame in place.

    Parameters
    ----------
    frame : MatLike
        The image (height × width × channels) on which to draw. Modified in place.
    text : str
        The string to be rendered on the image.
    org : Point
        The bottom-left corner of the text string in the image, as an (x, y) tuple.
    font_face : int, optional
        Font type, see OpenCV docs. Default is `cv2.FONT_HERSHEY_DUPLEX`.
    font_scale : float, optional
        Scale factor that is multiplied by the font-specific base size. Default is 0.9.
    color : Scalar, optional
        Text color in BGR order. Default is (0, 255, 0) (green).
    thickness : int, optional
        Thickness of the text strokes. Default is 2.

    Returns
    -------
    None
        The function draws the text directly on `frame` and returns nothing.
    """
    cv2.putText(frame, text, org, font_face, font_scale, color, thickness)


def draw_identified_box_ltrb(
    frame: cv2.typing.MatLike,
    text: str,
    left: int,
    top: int,
    right: int,
    bottom: int
) -> None:
    """
    Draws a labeled bounding box around a detected face (or object) using left, top,
    right, bottom coordinates, then overlays a filled label background and text.

    Parameters
    ----------
    frame : MatLike
        The image on which to draw (height × width × 3), modified in place.
    text : str
        The label to display (e.g., a person’s name or emotion).
    left : int
        X-coordinate of the top-left corner of the bounding box.
    top : int
        Y-coordinate of the top-left corner of the bounding box.
    right : int
        X-coordinate of the bottom-right corner of the bounding box.
    bottom : int
        Y-coordinate of the bottom-right corner of the bounding box.

    Returns
    -------
    None
        The function draws directly on `frame` and does not return anything.

    Details
    -------
    1. Draws an outer rectangle (the detection box) in the default color and thickness
       using `cv2_rectangle`.
    2. Draws a filled rectangle as a background for the label text just above the
       bottom edge of the box, using OpenCV’s `cv2.FILLED`.
    3. Renders the `text` inside that background area with white font, offset slightly
       from the left edge.
    """
    cv2_rectangle(frame, (left, top), (right, bottom))
    cv2.rectangle(frame, (left - 2, bottom), (right + 2, bottom + 35),
                   (255, 0, 0), cv2.FILLED)
    cv2_put_text(frame, text, (left + 4, bottom + 26),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                 2)


def draw_identified_box_xywh(
    frame: cv2.typing.MatLike,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int
) -> None:
    """
    Draws a labeled bounding box on an image using (x, y, width, height) coordinates.

    This function converts the rectangle specified by top-left corner (x, y)
    and dimensions (width, height) into left, top, right, bottom coordinates, then
    delegates to `draw_identified_box_ltrb` to render the box, background label,
    and text.

    Parameters
    ----------
    frame : MatLike
        The image on which to draw. This array is modified in place.
    text : str
        The label text to display (e.g., a person’s name or emotion).
    x : int
        X-coordinate of the top-left corner of the bounding box.
    y : int
        Y-coordinate of the top-left corner of the bounding box.
    width : int
        Width of the bounding box in pixels.
    height : int
        Height of the bounding box in pixels.

    Returns
    -------
    None
        The function draws directly onto `frame` and does not return a value.
    """
    draw_identified_box_ltrb(frame, text, x, y, (x + width), (y + height))


def detect_faces(
    bgr_frame: cv2.typing.MatLike,
    min_neighbors: int = 7,
    enable_dnn: bool   = True
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR image using either a DNN-based detector or a Haar cascade.

    This function provides a unified interface for face detection, allowing you to
    switch between a more accurate (but heavier) deep-neural-network approach and
    a lightweight Haar cascade classifier.

    Parameters
    ----------
    bgr_frame : MatLike
        The input image in BGR color space (height x width x 3), as typically
        returned by OpenCV’s `VideoCapture.read()` or `cv2.imread()`.
    min_neighbors : int, optional
        Only used when `enable_dnn` is False. Specifies the minimum number of
        adjacent detections a rectangle should have to be retained by the cascade
        (higher values yield fewer false positives). Default is 7.
    enable_dnn : bool, optional
        If True, uses the DNN-based face detector (`detect_faces_using_dnn`) for
        higher accuracy. If False, falls back to the cascade classifier
        (`detect_faces_using_cascade_classifier`). Default is True.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of face bounding boxes. Each box is a tuple `(x, y, w, h)` when using
        the cascade classifier, or `(left, top, right, bottom)` if returned by the DNN
        detector—ensure downstream code matches your chosen format.

    Notes
    -----
    - `detect_faces_using_dnn(bgr_frame)` should return a list of `(left, top, right, bottom)`.
    - `detect_faces_using_cascade_classifier(bgr_frame, min_neighbors)` should return
      a list of `(x, y, w, h)`.
    - Make sure your downstream drawing or encoding functions expect the correct box
      format for the chosen method.
    """
    return  detect_faces_using_dnn(bgr_frame) \
        if enable_dnn \
        else detect_faces_using_cascade_classifier(bgr_frame, min_neighbors)


def detect_faces_using_cascade_classifier(
    bgr_frame: cv2.typing.MatLike,
    min_neighbors: int = 7
) -> typing.Sequence[typing.Sequence[int]]:
    """
    Detect faces in a BGR image using OpenCV’s Haar cascade classifiers.

    This function applies both the frontal face and profile face cascades
    to detect faces in the image. It returns a combined list of bounding
    boxes from both detectors.

    Parameters
    ----------
    bgr_frame : MatLike
        Input image in BGR color space (height x width x 3), as typically
        provided by OpenCV methods like `cv2.imread` or `VideoCapture.read()`.
    min_neighbors : int, optional
        Parameter specifying how many neighbors each candidate rectangle
        should have to retain it. Higher values result in fewer detections
        with higher quality (fewer false positives). Default is 7.

    Returns
    -------
    typing.Sequence[typing.Sequence[int]]
        A list of face bounding boxes, each represented as (x, y, w, h),
        where (x, y) is the top-left corner, and w, h are the width and
        height of the detected face region.

    Notes
    -----
    - Uses `haarcascade_frontalface_default.xml` for frontal faces.
    - Uses `haarcascade_profileface.xml` for left/right profile faces.
    - Combines results from both detectors; overlapping boxes are not merged.
    """
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
        + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml')
    faces_frontal = face_cascade.detectMultiScale(
        image        = gray_frame,
        scaleFactor  = 1.15,
        minNeighbors = min_neighbors
    )
    faces_profile = profile_cascade.detectMultiScale(
        image        = gray_frame,
        scaleFactor  = 1.15,
        minNeighbors = min_neighbors
    )
    return list(faces_frontal) + list(faces_profile)


def detect_faces_using_dnn(
    bgr_frame: cv2.typing.MatLike,
    config_file: str = "./doc/model/deploy.prototxt",
    model_file: str  = "./doc/model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR image using OpenCV’s DNN-based SSD face detector.

    This function loads a Caffe model (ResNet-10 based SSD) and runs a forward
    pass to locate faces. It returns bounding boxes in (x, y, w, h) format.

    Parameters
    ----------
    bgr_frame : MatLike
        The input image in BGR color space (height x width x 3).
    config_file : str, optional
        Path to the Caffe "deploy" prototxt defining the network architecture.
    model_file : str, optional
        Path to the pretrained Caffe model weights (caffemodel) for face detection.

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of detected face bounding boxes. Each tuple is
        (x_min, y_min, width, height), where (x_min, y_min) is the
        top-left corner and width/height are the box dimensions.

    Notes
    -----
    - The network expects 300×300 pixel inputs in BGR order, with mean subtraction
      of (104.0, 177.0, 123.0).
    - Only detections with confidence > 0.5 are returned; you can adjust this
      threshold in the code for more or fewer detections.
    """
    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            bgr_frame,
            (300, 300)
        ),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net  = cv2.dnn.readNetFromCaffe(config_file, model_file)
    net.setInput(blob)
    detections = net.forward()
    h, w       = bgr_frame.shape[:2]

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return faces


def analyze_emotions(
    bgr_frame: cv2.typing.MatLike,
    faces: typing.Sequence[typing.Sequence[int]]
) -> list[str]:
    """
    Analyze the dominant emotion for each detected face region in a BGR image.

    This function takes an OpenCV BGR frame and a list of face bounding boxes,
    crops each face region, and uses DeepFace to predict the dominant emotion.
    It gracefully handles errors by labeling unanalyzable faces as "Unknown".

    Parameters
    ----------
    bgr_frame : MatLike
        The input image in BGR color space (height x width x 3).
    faces : Sequence of Sequence[int]
        A sequence of face bounding boxes, where each box is specified as
        (x, y, w, h). `x, y` are the top-left corner coordinates, and
        `w, h` are the width and height of the face region in pixels.

    Returns
    -------
    List[str]
        A list of detected emotions (e.g., "happy", "sad", "angry")—one per
        face in the input list. If emotion analysis fails for a face,
        "Unknown" is returned in its place.

    Notes
    -----
    - Uses `DeepFace.analyze` with:
        - `actions=['emotion']` to focus solely on emotion classification.
        - `detector_backend='centerface'` for robust face detection.
        - `enforce_detection=False` to proceed even if a face isn’t confidently detected.
        - `anti_spoofing=False` to disable liveness checks.
    - If DeepFace returns a single dict for one face, it is wrapped into a list
      for uniform processing.
    - Catches and logs exceptions, assigning "Unknown" on failure.
    """
    emotions = []
    for (x, y, w, h) in faces:
        face_img = bgr_frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(
                face_img,
                actions           = ['emotion'],
                # 'opencv', 'mtcnn'+, 'skip'++, 'mediapipe'++ or 'centerface'+++
                detector_backend  ='centerface',
                enforce_detection = False,
                anti_spoofing     = False
            )
            if isinstance(result, dict):
                result = [result]
            for face in result:
                if 'dominant_emotion' in face:
                    emotions.append(face['dominant_emotion'])
        except Exception as e:
            print(f"Failing analyzing emotions: {e}")
            emotions.append("Unknown")
    return emotions


def configure_pose_solutions() -> tuple[mp.solutions.pose, mp.solutions.pose.Pose]:
    """
    Configure and return MediaPipe Pose utilities for body landmark detection.

    This function sets up the MediaPipe pose solution module and creates
    an instance of the Pose class with default detection/tracking confidence
    thresholds.

    Returns
    -------
    tuple[mp.solutions.pose, mp.solutions.pose.Pose]
        - `mp_pose`: The MediaPipe pose module (for access to constants like POSE_CONNECTIONS).
        - `pose`: An initialized `mp_pose.Pose` object configured with:
            * `min_detection_confidence=0.5` — minimum confidence for the initial
              pose detection to be considered successful.
            * `min_tracking_confidence=0.5` — minimum confidence for subsequent
              landmark tracking to be considered reliable.
    """
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5
    )
    return mp_pose, pose


def configure_hands_solutions() \
    -> tuple[mp.solutions.hands, mp.solutions.hands.Hands]:
    """
    Configure and return MediaPipe Hands utilities for hand landmark detection.

    This function initializes the MediaPipe hands solution module and creates
    an instance of the Hands class with a higher-complexity model and tuned
    confidence thresholds.

    Returns
    -------
    Tuple[mp.solutions.hands, mp.solutions.hands.Hands]
        - `mp_hands`: The MediaPipe hands module (for access to constants like HAND_CONNECTIONS).
        - `hands`: An initialized `mp_hands.Hands` object configured with:
            * `model_complexity=1` — use the more detailed model (0 = lite, 1 = full).
            * `min_detection_confidence=0.7` — minimum confidence for initial hand detection.
            * `min_tracking_confidence=0.7` — minimum confidence for landmark tracking across frames.
            * `max_num_hands=4` — maximum number of hands to detect and track per frame.
    """
    mp_hands    = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity         = 1,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.7,
        max_num_hands            = 4
    )
    return mp_hands, hands


def configure_face_solutions() \
    -> tuple[mp.solutions.face_mesh, mp.solutions.face_mesh.FaceMesh]:
    """
    Configure and return MediaPipe FaceMesh utilities for facial landmark detection.

    This function initializes the MediaPipe face_mesh module and creates
    an instance of the FaceMesh class with default parameters for single-face
    detection.

    Returns
    -------
    Tuple[mp.solutions.face_mesh, mp.solutions.face_mesh.FaceMesh]
        - `mp_face`: The MediaPipe face_mesh module (for access to constants like FACEMESH_TESSELATION).
        - `face_mesh`: An initialized `mp_face.FaceMesh` object configured with:
            * `max_num_faces=1` — detect at most one face per frame.
            * `min_detection_confidence=0.5` — minimum confidence threshold for the initial face detection.
            * `min_tracking_confidence=0.5` — minimum confidence threshold for tracking facial landmarks across frames.
    """
    mp_face     = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces            = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5
    )
    return mp_face, face_mesh
