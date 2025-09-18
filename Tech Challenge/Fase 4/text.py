from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from tqdm import tqdm

import csv
import speech_recognition as sr


def transcribe_sr_audio(sr_audio, text_file, use_google: bool = True) -> None:
    """
    Transcribe speech from an AudioData object and save the text to a file.

    This function uses the SpeechRecognition library to perform speech-to-text
    on an AudioData instance, either via Google’s free web API or via the Vosk
    offline recognizer. It then writes the resulting transcription to disk
    in manageable chunks, showing progress as it saves.

    Parameters
    ----------
    sr_audio : sr.AudioData
        An AudioData object containing the audio samples and sample rate,
        typically obtained via `recognizer.record(source)` on an `sr.AudioFile`.
    text_file : str
        Path (including filename) where the transcription should be saved.
        The file will be created or overwritten.
    use_google : bool, optional
        If True, use `recognize_google` (requires internet). If False,
        use `recognize_vosk` (offline, requires Vosk model installed).
        Default is True.

    Returns
    -------
    None
        Writes the transcription to `text_file` and prints it to the console.
        Does not return a value.

    Raises
    ------
    sr.RequestError
        If the Google API is unreachable or returns an error.
    sr.UnknownValueError
        If the recognizer could not understand the audio.
    Exception
        Any other errors during recognition or file I/O are propagated.

    Side Effects
    ------------
    - Prints the full transcription to stdout.
    - Displays a tqdm progress bar while writing the file in 1 KB chunks.
    """
    recognizer = sr.Recognizer()

    if use_google:
        text = recognizer.recognize_google(sr_audio, language="en-US")
    else:
        text = recognizer.recognize_vosk(sr_audio, language="en-US")
    print(f"\nTranscription: {text}\n" )

    with open(text_file, 'w', encoding='utf-8') as file:
        chunk_size = 1024  # 1KB por chunk
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in tqdm(text_chunks, desc="Saving transcription to file"):
            file.write(chunk)


def ensure_file_exists(path: str) -> None:
    """
    Ensure that a file at the given path exists.
    If it does not exist, create an empty file.

    Parameters
    ----------
    path : str
        The filesystem path to the file to check or create.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)


def save_results_to_csv(
    results: Mapping[Any, Sequence[Any]],
    output_path: str,
    headers: Sequence[str] = ("frame_id", "person_name")
) -> None:
    """
    Save a list of (int, str) tuples to a CSV file.

    Parameters
    ----------
    results : Mapping[Any, Sequence[Any]]
        A mapping from each key to a sequence of one-or-more values.
    output_path : str
        The file path where the CSV will be saved.
    headers : Sequence[str], optional
        Column headers; defaults to ("id", "value").

    Returns
    -------
    None
        This function performs file I/O and does not return a value.
    """
    ensure_file_exists(output_path)
    max_values = max(
        (len(vals) for vals in results.values()),
        default = 0
    )
    with open(output_path, "w", newline = "", encoding = "utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for key, vals in results.items():
            pad_len = max_values - 1 - len(vals)
            row = [key] + list(vals) + [""] * pad_len
            writer.writerow(row)


def count_appearances(
    csv_path: str,
    pause_threshold: int = 30,
    min_segment_length: int = 5
) -> dict[str, int]:
    """
    Count discrete valid appearances of each person based on frame-by-frame detections.

    An appearance is defined as a continuous segment of detections for a given person
    where:
      - Any gap of more than `pause_threshold` frames marks the end of one segment
        and the start of the next.
      - Only segments with at least `min_segment_length` consecutive frames are
        counted as valid appearances.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing detection data. The file is expected to
        have a header row, then rows of the form:
            frame_id,person_name
        Rows with an empty name are ignored.
    pause_threshold : int, optional
        The minimum number of frames of “no detection” required before treating
        subsequent detections as a new appearance. Default is 30.
    min_segment_length : int, optional
        The minimum number of consecutive frames for a detection segment to be
        considered a valid appearance. Shorter segments are treated as false
        positives/noise. Default is 3.

    Returns
    -------
    dict[str, int]
        A mapping from each person’s name to the number of valid appearances
        detected in the video.
    """
    # Group all detected frames by person
    frames_by_person: dict[str, list[int]] = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row

        for row in reader:
            if not row or not row[0].strip():
                continue  # skip empty lines or missing frame IDs
            frame = int(row[0])
            name  = row[1].strip() if len(row) > 1 else ''
            if name:
                frames_by_person[name].append(frame)

    # For each person, sort their frames and count valid segments
    appearances: dict[str, int] = {}
    for name, frames in frames_by_person.items():
        frames.sort()
        valid_count = 0

        # Initialize first segment
        prev_frame = frames[0]
        segment_length = 1

        # Iterate over subsequent detections
        for current in frames[1:]:
            if current - prev_frame <= pause_threshold:
                # Still the same appearance segment
                segment_length += 1
            else:
                # Gap too large: end previous segment
                if segment_length >= min_segment_length:
                    valid_count += 1
                # Start a new segment
                segment_length = 1
            prev_frame = current

        # Count the last segment if it’s long enough
        if segment_length >= min_segment_length:
            valid_count += 1

        appearances[name] = valid_count

    return appearances


def count_emotion_appearances(
    csv_path: str,
    pause_threshold: int = 30,
    min_segment_length: int = 1
) -> dict[str, int]:
    """
    Count discrete appearance segments for each emotion in a multi-column CSV.

    An “appearance” of emotion X is any continuous run of frames where X
    appears in at least one of the emotion columns, such that:
      - A gap > pause_threshold frames breaks the run.
      - Only runs of length >= min_segment_length are counted.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. First row is header, subsequent rows:
            frame_id,emotion1,emotion2,emotion3,...
        Blank cells are ignored.
    pause_threshold : int, optional
        Number of frames of absence that must occur before a new appearance
        is counted. Default is 30.
    min_segment_length : int, optional
        Minimum consecutive frames in which the emotion must be detected
        to count as one appearance. Default is 1.

    Returns
    -------
    dict[str, int]
        Mapping from emotion name to number of valid appearances.
    """
    # 1) Collect frames per emotion
    frames_by_emotion: dict[str, list[int]] = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if not row or not row[0].strip():
                continue
            frame = int(row[0])
            # all emotion columns
            for emotion in row[1:]:
                emo = emotion.strip()
                if emo:
                    frames_by_emotion[emo].append(frame)

    # 2) For each emotion, sort frames and count segments
    appearance_counts: dict[str, int] = {}

    for emo, frames in frames_by_emotion.items():
        frames.sort()
        count = 0

        # initialize the first segment
        prev_frame = frames[0]
        seg_len = 1

        # iterate through subsequent frames
        for fr in frames[1:]:
            if fr - prev_frame <= pause_threshold:
                seg_len += 1
            else:
                # segment break: count it if long enough
                if seg_len >= min_segment_length:
                    count += 1
                seg_len = 1
            prev_frame = fr

        # count the last segment
        if seg_len >= min_segment_length:
            count += 1

        appearance_counts[emo] = count

    return appearance_counts
