from arch_wise.architecture_detection_service import ArchitectureDetectionService
from arch_wise.config import DEFAULT_SOURCE_FILE_PATH

if __name__ == '__main__':
    # Initialize detection service with default parameters
    architecture_detection_service = ArchitectureDetectionService()
    
    # Process the image (detect architecture components and generate report)
    results, report_json, markdown_report = architecture_detection_service.process_image(DEFAULT_SOURCE_FILE_PATH)
    
    print(f"✅ Architecture detection completed with {len(results)} detections")
    print(f"✅ Report generated successfully")
    print(f"📄 Report content length: {len(markdown_report)} characters")
