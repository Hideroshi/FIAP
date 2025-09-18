from moviepy import VideoFileClip
from tqdm import tqdm

import cv2


def cv2_video_capture(input_video_file: str | int = 0) -> cv2.VideoCapture:
    """
    Open a video source (file or camera) and verify it was successfully initialized.

    This helper wraps `cv2.VideoCapture`, providing a clear error if the
    capture cannot be opened.

    Parameters
    ----------
    input_video_file : str or int, optional
        The path to a video file (e.g. "video.mp4") or an integer camera index
        (e.g. 0 for the default webcam). Default is 0 (first camera).

    Returns
    -------
    cv2.VideoCapture
        An OpenCV VideoCapture object, ready for `.read()` calls.

    Raises
    ------
    Exception
        If the video source cannot be opened, an exception is raised with a
        descriptive message.
    """
    capture = cv2.VideoCapture(input_video_file)
    if not capture.isOpened():
       raise Exception("Video capture could not be opened!")
    return capture


def read_frame(capture:  cv2.VideoCapture) -> cv2.typing.MatLike:
    """
    Read a single frame from an OpenCV VideoCapture object and validate the result.

    This helper wraps `capture.read()` to provide a clear error if the frame
    cannot be retrieved (e.g., end of stream or camera error).

    Parameters
    ----------
    capture : cv2.VideoCapture
        An already opened OpenCV VideoCapture instance.

    Returns
    -------
    MatLike
        The image frame read from the capture, as a NumPy array.

    Raises
    ------
    Exception
        If the frame could not be read (`ret` is False), indicating the video
        has ended or there was an I/O error.
    """
    ret, frame = capture.read()
    if not ret:
       raise Exception("Frame could not be read!")
    return frame


def break_video_into_indexed_frames(
    capture: cv2.VideoCapture
) -> list[tuple[int, cv2.typing.MatLike]]:
    """
    Read all frames from an open VideoCapture and return them with their frame indices.

    This function queries the total frame count, then iterates through the video,
    reading each frame sequentially. It pairs each frame with its zero-based index
    in the video and returns the list of (index, frame) tuples.

    Parameters
    ----------
    capture : cv2.VideoCapture
        An already-opened OpenCV VideoCapture instance, positioned at the first frame.
        After calling this function, the capture will be at the end of the stream.

    Returns
    -------
    list[tuple[int, MatLike]]
        A list where each element is a tuple `(i, frame)`, with `i` being the
        frame number (starting at 0) and `frame` the BGR image array.

    Raises
    ------
    Exception
        If any frame cannot be read (e.g., I/O error or unexpected end of video),
        the helper `read_frame` will raise an Exception.

    Notes
    -----
    - This will load the entire video into memory; for very large videos, consider
      processing frame-by-frame without storing all frames at once.
    - The capture’s internal pointer will be at the end of the video after this call.
    """
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in tqdm(range(total_frames), desc = "Indexing video frames"):
        frame = read_frame(capture)
        frames.append((i, frame))
    return frames


def retrieve_video_capture_properties(
    capture: cv2.VideoCapture
) -> tuple[int, int, int, int, int]:
    """
    Retrieve fundamental properties from an OpenCV VideoCapture object.

    This helper reads the frame width, frame height, frames per second (FPS),
    and total frame count directly from the capture. It also constructs a FOURCC
    code for MP4 output (using "mp4v").

    Parameters
    ----------
    capture : cv2.VideoCapture
        An already-opened VideoCapture instance (e.g., from `cv2.VideoCapture("file.mp4")`).

    Returns
    -------
    Tuple[int, int, int, int, int]
        A 5-tuple containing:
        - width (int):  The width of each video frame in pixels.
        - height (int): The height of each video frame in pixels.
        - fps (int):    The capture’s frames-per-second rate (rounded to nearest int).
        - total_frames (int): The total number of frames in the video (if known).
        - fourcc (int): The FOURCC code for writing MP4 files (cv2.VideoWriter_fourcc(*"mp4v")).
    """
    width        = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v') # Codec para MP4
    return width, height, fps, total_frames, fourcc


def cv2_video_writer(
    video_capture: cv2.VideoCapture,
    output_path: str
) -> cv2.VideoWriter:
    """
    Create an OpenCV VideoWriter configured to match the properties of an existing VideoCapture.

    This helper reads the width, height, FPS, and FOURCC codec from the provided
    VideoCapture and uses them to initialize a VideoWriter for writing out video
    frames that align with the input source.

    Parameters
    ----------
    video_capture : cv2.VideoCapture
        An already-opened OpenCV VideoCapture instance (e.g., from `cv2_video_capture`).
    output_path : str
        Path where the output video file will be saved (including extension, e.g. ".mp4").

    Returns
    -------
    cv2.VideoWriter
        An OpenCV VideoWriter object ready to accept frames via `writer.write(frame)`.
    """
    width, height, fps, _, fourcc = retrieve_video_capture_properties(video_capture)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def load_video_file_clip_from_path(video_file: str) -> VideoFileClip:
    """
    Load a video file into a MoviePy VideoFileClip with a progress bar.

    Uses a simple tqdm progress bar to indicate that the file is being loaded,
    which is useful for large files or slower I/O.

    Parameters
    ----------
    video_file : str
        Path to the video file to load (e.g., ".mp4", ".mov").

    Returns
    -------
    VideoFileClip
        A MoviePy VideoFileClip object representing the loaded video,
        which can be used for further audio/video processing.
    """
    with tqdm(total=1, desc="Loading video file") as pbar:
        video = VideoFileClip(video_file)
        pbar.update(1)
    return video
