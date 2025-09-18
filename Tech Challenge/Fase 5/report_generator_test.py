from report_generator import generate_report
from model_predict import predict

if __name__ == '__main__':
    trained_dir_name: str = "yolo11s_custom_100"
    trained_model_best_path = f"./data/model/trained/{trained_dir_name}/weights/best.pt"
    source_file_path: str = "./data/sample/aws_01.jpg"

    reports_path: str = f"data/reports/{trained_dir_name}"
    #reports_json_path: str = f"{reports_path}/report.json"
    stride_yaml_path  = "data/stride/stride_map.yaml"
    controls_yaml= "data/stride/controls_by_threat.yaml"
    conf = 0.7

    results, report_json = predict(
        trained_dir_name,
        trained_model_best_path,
        source_file_path,
        conf
    )
    generate_report(
        # reports_json_path = reports_json_path,
        reports_json_path = report_json,
        stride_yaml_path  = stride_yaml_path,
        controls_yaml     = controls_yaml,
        conf_threshold    = 0.2,
        reports_path      = reports_path  # opcional: aumenta organização
    )