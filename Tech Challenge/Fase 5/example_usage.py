"""
Example usage of the modular architecture detection components.
This demonstrates how the modular design makes the code reusable.
"""

from arch_wise.architecture_detection_service import ArchitectureDetectionService
from arch_wise.temp_file_handler import TempFileHandler
from arch_wise.config import DEFAULT_SOURCE_FILE_PATH

def example_1_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize service with default parameters
    service = ArchitectureDetectionService()
    
    # Process an image
    results, report_json, markdown_report = service.process_image(DEFAULT_SOURCE_FILE_PATH)
    
    print(f"Detected {len(results)} architecture components")
    print(f"Report JSON items: {len(report_json)}")
    print(f"Report content length: {len(markdown_report)} characters")
    print("✅ Basic usage completed\n")

def example_2_custom_parameters():
    """Example 2: Usage with custom parameters."""
    print("=== Example 2: Custom Parameters ===")
    
    # Initialize service with custom parameters
    service = ArchitectureDetectionService(
        trained_dir_name="yolo11n_custom_100",
        conf=0.8  # Higher confidence threshold
    )
    
    # Process an image
    results, report_json, markdown_report = service.process_image(DEFAULT_SOURCE_FILE_PATH)
    
    print(f"Detected {len(results)} architecture components with custom confidence threshold")
    print(f"Report JSON items: {len(report_json)}")
    print(f"Report content length: {len(markdown_report)} characters")
    print("✅ Custom parameters usage completed\n")

def example_3_step_by_step():
    """Example 3: Step-by-step processing."""
    print("=== Example 3: Step-by-Step Processing ===")
    
    service = ArchitectureDetectionService()
    
    # Step 1: Detect architecture components only
    results, report_json = service.detect_architecture_components(DEFAULT_SOURCE_FILE_PATH)
    print(f"Step 1: Detected {len(results)} architecture components")
    
    # Step 2: Generate report separately
    markdown_report = service.generate_detection_report(report_json)
    print(f"Step 2: Report generated with {len(markdown_report)} characters")
    
    print("✅ Step-by-step processing completed\n")

def example_4_temp_file_handling():
    """Example 4: Temporary file handling."""
    print("=== Example 4: Temporary File Handling ===")
    
    # Simulate saving a file to temp location
    temp_path = TempFileHandler.save_image_to_temp(DEFAULT_SOURCE_FILE_PATH)
    print(f"Saved image to temporary location: {temp_path}")
    
    # Process the temp file
    service = ArchitectureDetectionService()
    results, report_json, markdown_report = service.process_image(temp_path)
    
    # Clean up
    TempFileHandler.cleanup_temp_file(temp_path)
    print("Cleaned up temporary file")
    
    print(f"✅ Processed {len(results)} detections from temp file")
    print(f"📄 Report JSON items: {len(report_json)}")
    print(f"📄 Report content length: {len(markdown_report)} characters\n")

if __name__ == '__main__':
    # Run all examples
    example_1_basic_usage()
    example_2_custom_parameters()
    example_3_step_by_step()
    example_4_temp_file_handling()
    
    print("🎉 All examples completed successfully!")
