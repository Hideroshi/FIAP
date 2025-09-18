from flask import Flask, request, jsonify
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arch_wise.temp_file_handler import TempFileHandler
from arch_wise.architecture_detection_service import ArchitectureDetectionService
from arch_wise.auth import require_api_key

app = Flask(__name__)

"""
    server for architecture detection.
    This server is used to detect architecture components in an image.
    It uses the ArchitectureDetectionService to detect the components and generate a report.
    It also uses the TempFileHandler to save the uploaded image to a temporary file.
    It then cleans up the temporary file after the detection is complete.

    Args:
        image: The image to detect the architecture components in.

    Returns:
        A JSON object with the following fields:
        - success: A success message.
        - detections: The number of architecture components detected.
        - report_generated: A boolean indicating if the report was generated.
        - markdown_report: The complete markdown report content for the Telegram agent.
        - error: An error message if the detection fails.

    Example:
        curl -X POST -F "image=@path/to/image.jpg" -H "X-API-Key: your_api_key" http://localhost:8000/architecture-detect

    Example response:
        {
            "success": "Architecture detection completed successfully!",
            "detections": 10,
            "report_generated": true,
            "markdown_report": "# Threat Model Report\n\nGerado automaticamente\n\n## Component A (image.jpg)\n- **Confiança:** 0.85\n..."
        }
"""
@app.route('/architecture-detect', methods=['POST'])
@require_api_key
def architecture_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'An image is required'}), 400
    
    # Get the uploaded image
    image_file = request.files['image']

    #Get the chat details
    chat_details = request.form.get('chat_details')
    if not chat_details:
        return jsonify({'error': 'Chat details are required'}), 400
    
    #Parse the chat details
    chat_details = json.loads(chat_details)
    
    # Save the uploaded image to a temporary file
    temp_image_path = TempFileHandler.save_uploaded_file(image_file)
    
    try:
        # Initialize architecture detection service
        architecture_detection_service = ArchitectureDetectionService()
        
        # Process the image (detect architecture components and generate report)
        results, report_json, markdown_report = architecture_detection_service.process_image(temp_image_path)
        
        return jsonify({
            'success': 'Architecture detection completed successfully!',
            'detections': len(results),
            'report_generated': True,
            'markdown_report': markdown_report,
            'chat_details': chat_details
        }), 201
        
    except Exception as detection_error:
        print(f"❌ Architecture detection failed: {str(detection_error)}")
        return jsonify({'error': f'Architecture detection failed'}), 500
    
    finally:
        # Clean up the temporary file
        TempFileHandler.cleanup_temp_file(temp_image_path)

if __name__ == '__main__':
    # Using Flask's built-in development server
    # For production, use: gunicorn -w 4 -b 0.0.0.0:8000 server:app
    app.run(host='0.0.0.0', port=8000, debug=True)
