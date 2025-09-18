import tempfile
import os
from typing import Optional
from PIL import Image


class TempFileHandler:
    """Utility class for handling temporary file operations."""
    
    @staticmethod
    def save_uploaded_file(uploaded_file, suffix: str = '.jpg') -> str:
        """
        Save an uploaded file to a temporary location.
        
        Args:
            uploaded_file: The uploaded file object from Flask
            suffix (str): File extension for the temporary file
            
        Returns:
            str: Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            return temp_file.name
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """
        Clean up a temporary file.
        
        Args:
            file_path (str): Path to the temporary file to delete
        """
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    
    @staticmethod
    def save_image_to_temp(image_path: str, suffix: str = '.jpg') -> str:
        """
        Save an image file to a temporary location.
        
        Args:
            image_path (str): Path to the source image file
            suffix (str): File extension for the temporary file
            
        Returns:
            str: Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            with open(image_path, 'rb') as source_file:
                temp_file.write(source_file.read())
            return temp_file.name 