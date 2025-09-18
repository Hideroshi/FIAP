from moviepy import VideoFileClip, AudioFileClip

import speech_recognition as sr


def get_audio(video_file_clip: VideoFileClip) -> AudioFileClip:
    """
    Extract the audio track from a video clip.
    Parameters
    ----------
    video_file_clip: VideoFileClip
        An instance of MoviePy’s VideoFileClip representing the input video.
    Returns
    -------
    AudioFileClip
        The audio track of the provided video clip, as a MoviePy AudioFileClip.
    """
    return video_file_clip.audio


def write_audio_to_path(
        video_file_clip: VideoFileClip,
        audio_out_path: str
) -> None:
    """
    Extracts the audio track from a VideoFileClip and writes it out to a file.

    Parameters
    ----------
    video_file_clip: VideoFileClip
        A MoviePy VideoFileClip object representing the source video.
    audio_out_path: str
        The path (including filename and extension) where the extracted audio
        should be saved. Supported formats depend on ffmpeg (e.g. “.wav”, “.mp3”).
    Returns
    -------
    None
        This function writes the audio file to disk and does not return a value.

    Side Effects
    ------------
    - Prints a status message to the console.
    - Creates or overwrites the file at `audio_file`.

    Notes
    -----
    - Uses a sampling rate of 44100 Hz (CD quality).
    - Outputs 16-bit, mono audio by default.
    - You can adjust parameters like `fps`, `nbytes`, `codec`, and `ffmpeg_params`
      directly in the call to `AudioFileClip.write_audiofile` if you need a different format.
    """
    print("\nLoading audio from video clip file...")
    audio = get_audio(video_file_clip)
    audio.write_audiofile(
        audio_out_path,
        fps           = 44100,        # Standard sampling rate for CD-quality audio
        nbytes        = 2,            # Bit depth (2 = 16-bit, 4 = 32-bit)
        # Uncompressed codec for maximum compatibility: use 'pcm_s16le' for
        codec         = 'pcm_s16le',  # 16-bit WAV or 'pcm_s32le' for 32-bit WAV
        ffmpeg_params = ["-ac", "1"], # Force mono output
        #logger        = None          # Disable additional logging
    )


def load_sr_audio_from_path(audio_out_path: str) -> sr.AudioData:
    """
    Load an audio file and convert it into an AudioData object for speech recognition.

    This function uses the SpeechRecognition library to read the entire contents
    of the specified audio file into memory, returning an AudioData instance that
    can be passed to recognizer methods like `recognize_google`.

    Parameters
    ----------
    audio_out_path : str
        Path to the audio file on disk. Supported formats depend on your ffmpeg
        or native WAV support, e.g. “.wav”, “.flac”, “.mp3”, etc.

    Returns
    -------
    sr.AudioData
        An AudioData object containing the raw audio samples and sample rate.

    Side Effects
    ------------
    - Prints a status message to the console.
    - Reads the entire audio file into memory.

    Raises
    ------
    FileNotFoundError
        If the file at `audio_out_path` does not exist.
    sr.AudioFileError
        If the file cannot be parsed as audio.
    """
    print("\nLoading audio from file...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_out_path) as source:
        sr_audio = recognizer.record(source)
    return sr_audio
