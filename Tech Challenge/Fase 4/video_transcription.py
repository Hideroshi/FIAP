from audio import write_audio_to_path, load_sr_audio_from_path
from text import transcribe_sr_audio
from video import load_video_file_clip_from_path

def execute_video_transcription(
    video_in_path: str,
    audio_out_path: str,
    text_out_path: str
) -> None:
    """
    Execute end-to-end transcription of a video file:
      1. Load the video.
      2. Extract and save its audio track.
      3. Load the audio into a SpeechRecognition AudioData object.
      4. Perform speech-to-text transcription (using Vosk offline).
      5. Write the resulting transcript to a text file.

    Parameters
    ----------
    video_in_path : str
        Path to the input video file (e.g., "input.mp4").
    audio_out_path : str
        Path where the extracted audio WAV will be saved (e.g., "audio.wav").
    text_out_path : str
        Path where the final transcription text will be written (e.g., "transcript.txt").

    Returns
    -------
    None
        Produces two files on disk (audio_out_path and text_out_path) and prints
        progress/status messages to the console.

    Raises
    ------
    Exception
        If any step fails (e.g., video cannot be loaded, audio extraction fails,
        audio file cannot be read, or transcription engine errors).
    """
    video = load_video_file_clip_from_path(video_in_path)
    write_audio_to_path(video, audio_out_path)
    sr_audio = load_sr_audio_from_path(audio_out_path)
    transcribe_sr_audio(sr_audio, text_out_path, False)

# Running main function
if __name__ == "__main__":
    video_in_path  = './doc/videos/tc4_video.mp4'
    audio_out_path = './doc/videos/result/tc4_audio.wav'
    text_out_path  = './doc/videos/result/tc4_video_transcription.txt'
    execute_video_transcription(video_in_path, audio_out_path, text_out_path)