"""
Authentication utilities for the architecture detection API.
"""

import os
from functools import wraps
from flask import request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv('API_KEY')

def require_api_key(f):
    """
    Decorator to require API key authentication.
    
    This decorator checks for the presence of a valid API key in the request headers.
    The API key should be provided in the 'X-API-Key' header.
    
    Args:
        f: The function to decorate
        
    Returns:
        The decorated function that checks for API key authentication
        
    Raises:
        401: If no API key is provided or if the API key is invalid
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from request headers
        api_key = request.headers.get('X-API-Key')
        
        # Check if API key is provided and valid
        if not api_key:
            return jsonify({'error': 'API key is required'}), 401
        
        if api_key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function
