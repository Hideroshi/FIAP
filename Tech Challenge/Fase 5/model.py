from pprint import pprint

from model_predict import predict
from model_train import train


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    yolo: str             = 'yolo11s'
    epochs: int           = 100
    yolo_pt_path: str     = f'./data/model/{yolo}.pt'
    data_yaml_path: str   = './data/dataset/aws/data.yaml'
    trained_dir_name: str = f'{yolo}_custom_{epochs}'

    trained_model_best_path = train(
        yolo_pt_path,
        data_yaml_path,
        trained_dir_name,
        epochs
    )
    # source_file_path: str = "./data/sample/aws_01.jpg"
    source_file_path: str = "./data/sample/aws_02.png"
    conf                  = 0.7

    results = predict(
        trained_dir_name,
        trained_model_best_path,
        source_file_path,
        conf
    )
    pprint(results)
