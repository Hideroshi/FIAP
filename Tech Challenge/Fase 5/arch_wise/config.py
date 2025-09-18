"""
Configuration module for architecture detection application.
Centralizes all configuration parameters to avoid duplication.
"""

# Model Configuration
DEFAULT_TRAINED_DIR_NAME = "yolo11n_custom_100"
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_REPORT_CONF_THRESHOLD = 0.2

# File Paths
DEFAULT_SOURCE_FILE_PATH = "./data/sample/aws_01.jpg"
STRIDE_YAML_PATH = "data/stride/stride_map.yaml"
CONTROLS_YAML_PATH = "data/stride/controls_by_threat.yaml"

# Model Paths
def get_model_path(trained_dir_name: str = DEFAULT_TRAINED_DIR_NAME) -> str:
    """Get the path to the trained model weights file."""
    return f"./data/model/trained/{trained_dir_name}/weights/best.pt"

def get_reports_path(trained_dir_name: str = DEFAULT_TRAINED_DIR_NAME) -> str:
    """Get the path to the reports directory."""
    return f"data/reports/{trained_dir_name}"
