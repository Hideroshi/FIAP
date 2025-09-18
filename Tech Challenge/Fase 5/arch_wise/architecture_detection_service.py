import os
from typing import Tuple, List, Any
from model_predict import predict
from report_generator import generate_report
from arch_wise.config import (
    DEFAULT_TRAINED_DIR_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_REPORT_CONF_THRESHOLD,
    STRIDE_YAML_PATH,
    CONTROLS_YAML_PATH,
    get_model_path,
    get_reports_path
)

class ArchitectureDetectionService:
    """Service class for architecture component detection operations."""
    
    def __init__(self, trained_dir_name: str = DEFAULT_TRAINED_DIR_NAME, conf: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the architecture detection service.
        
        Args:
            trained_dir_name (str): Name of the trained model directory
            conf (float): Confidence threshold for detections
        """
        self.trained_dir_name = trained_dir_name
        self.conf = conf
        self.trained_model_best_path = get_model_path(trained_dir_name)
        self.reports_path = get_reports_path(trained_dir_name)
        self.stride_yaml_path = STRIDE_YAML_PATH
        self.controls_yaml = CONTROLS_YAML_PATH
    
    def detect_architecture_components(self, image_path: str) -> Tuple[List[Any], List[Any]]:
        """
        Perform architecture component detection on an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[List[Any], List[Any]]: Detection results and report JSON
        """
        results, report_json = predict(
            self.trained_dir_name,
            self.trained_model_best_path,
            image_path,
            self.conf
        )
        return results, report_json
    
    def generate_detection_report(self, report_json: List[Any], conf_threshold: float = DEFAULT_REPORT_CONF_THRESHOLD) -> str:
        """
        Generate a report from detection results.
        
        Args:
            report_json (List[Any]): Detection results in JSON format
            conf_threshold (float): Confidence threshold for report generation
            
        Returns:
            str: The markdown content of the generated report
        """
        # Generate the report (this creates both .md and .html files)
        generate_report(
            reports_json_path=report_json,
            stride_yaml_path=self.stride_yaml_path,
            controls_yaml=self.controls_yaml,
            conf_threshold=conf_threshold,
            reports_path=self.reports_path
        )
        
        # Read and return the markdown content
        md_path = os.path.join(self.reports_path, "report.md")
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "Report generation completed but markdown file not found."
    
    def process_image(self, image_path: str) -> Tuple[List[Any], List[Any], str]:
        """
        Complete image processing pipeline: detect architecture components and generate report.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[List[Any], List[Any], str]: Detection results, report JSON, and markdown report
        """
        # Detect architecture components
        results, report_json = self.detect_architecture_components(image_path)
        
        # Generate report and get markdown content
        markdown_report = self.generate_detection_report(report_json)
        
        return results, report_json, markdown_report
