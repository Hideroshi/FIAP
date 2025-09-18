from face_detection_recognition import face_detection_and_recognition
from face_expression import face_expression
from pose_activity import detect_pose_and_activities
from video_transcription import execute_video_transcription

if __name__ == "__main__":

    images_path    = "./doc/images"
    video_in_path  = "./doc/videos/tc4_video.mp4"
    video_out_path = "./doc/videos/result/tc4_video_fr.mp4"
    face_detection_and_recognition(images_path, video_in_path, video_out_path)

    video_out_path = "./doc/videos/result/tc4_video_fe.mp4"
    face_expression(video_in_path, video_out_path)

    video_out_path = "./doc/videos/result/tc4_video_pa.mp4"
    detect_pose_and_activities(video_in_path, video_out_path)

    audio_out_path = './doc/videos/result/tc4_audio.wav'
    text_out_path  = './doc/videos/result/tc4_video_transcription.txt'
    execute_video_transcription(video_in_path, audio_out_path, text_out_path)