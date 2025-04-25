"""
Enhanced metadata extraction module that uses the optimized API client.
"""

import streamlit as st
import logging
import json
import re
import os
import time
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metadata_freeform(file_id: str, prompt: str, model: str = "google__gemini_2_0_flash_001") -> Dict[str, Any]:
    """
    Extract metadata from a file using freeform prompt
    
    Args:
        file_id: Box file ID
        prompt: Prompt for metadata extraction
        model: AI model to use
        
    Returns:
        dict: Extracted metadata
    """
    # Use enhanced client if available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        try:
            return st.session_state.enhanced_client.extract_metadata_freeform(file_id, prompt, model)
        except Exception as e:
            logger.error(f"Error using enhanced client for metadata extraction: {str(e)}")
            # Fall back to original implementation
    
    # Original implementation for backward compatibility
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",  # Required parameter - single_item_qa or multiple_item_qa
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"  # Required parameter for basic_text
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making Box AI API call with request: {json.dumps(request_body)}")
        import requests
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Log response for debugging
        logger.info(f"Box AI API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        logger.info(f"Box AI API response data: {json.dumps(response_data)}")
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the answer to extract key-value pairs
            metadata = parse_freeform_response(answer_text)
            
            return {
                "metadata": metadata,
                "raw_response": answer_text
            }
        
        # If no answer in response, return empty metadata
        return {
            "metadata": {},
            "raw_response": "No response from AI"
        }
    
    except Exception as e:
        logger.error(f"Error in Box AI API call: {str(e)}")
        raise Exception(f"Error extracting metadata: {str(e)}")

def extract_metadata_structured(file_id: str, fields: List[Dict[str, str]], model: str = "google__gemini_2_0_flash_001") -> Dict[str, Any]:
    """
    Extract structured metadata from a file
    
    Args:
        file_id: Box file ID
        fields: List of fields to extract with name and description
        model: AI model to use
        
    Returns:
        dict: Extracted metadata
    """
    # Use enhanced client if available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        try:
            return st.session_state.enhanced_client.extract_metadata_structured(file_id, fields, model)
        except Exception as e:
            logger.error(f"Error using enhanced client for structured metadata extraction: {str(e)}")
            # Fall back to original implementation
    
    # Original implementation for backward compatibility
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Create prompt with fields
    fields_text = "\n".join([f"- {field['name']}: {field['description']}" for field in fields])
    
    prompt = (
        f"Extract the following metadata fields from this document:\n\n"
        f"{fields_text}\n\n"
        f"Provide your answer in JSON format with field names as keys and extracted values as values. "
        f"If a field is not found in the document, set its value to null."
    )
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",  # Required parameter - single_item_qa or multiple_item_qa
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"  # Required parameter for basic_text
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making Box AI API call with request: {json.dumps(request_body)}")
        import requests
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Log response for debugging
        logger.info(f"Box AI API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        logger.info(f"Box AI API response data: {json.dumps(response_data)}")
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the answer to extract JSON
            metadata = parse_structured_response(answer_text, [field["name"] for field in fields])
            
            return {
                "metadata": metadata,
                "raw_response": answer_text
            }
        
        # If no answer in response, return empty metadata
        return {
            "metadata": {},
            "raw_response": "No response from AI"
        }
    
    except Exception as e:
        logger.error(f"Error in Box AI API call: {str(e)}")
        raise Exception(f"Error extracting metadata: {str(e)}")

def extract_metadata_template(file_id: str, template_id: str, model: str = "google__gemini_2_0_flash_001") -> Dict[str, Any]:
    """
    Extract metadata from a file using a metadata template
    
    Args:
        file_id: Box file ID
        template_id: Metadata template ID
        model: AI model to use
        
    Returns:
        dict: Extracted metadata
    """
    # Use enhanced client if available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        try:
            return st.session_state.enhanced_client.extract_metadata_template(file_id, template_id, model)
        except Exception as e:
            logger.error(f"Error using enhanced client for template metadata extraction: {str(e)}")
            # Fall back to original implementation
    
    # Get template fields
    if not st.session_state.metadata_templates or template_id not in st.session_state.metadata_templates:
        raise ValueError(f"Template {template_id} not found in session state")
    
    template = st.session_state.metadata_templates[template_id]
    
    # Convert template fields to structured fields
    fields = []
    for field in template["fields"]:
        fields.append({
            "name": field["key"],
            "description": field.get("displayName", field["key"])
        })
    
    # Use structured extraction
    return extract_metadata_structured(file_id, fields, model)

def parse_freeform_response(response_text: str) -> Dict[str, str]:
    """
    Parse freeform response to extract key-value pairs
    
    Args:
        response_text: AI response text
        
    Returns:
        dict: Extracted key-value pairs
    """
    metadata = {}
    
    try:
        # Try to parse as JSON first
        try:
            # Check if the response contains a JSON object
            json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                metadata = json.loads(json_str)
                return metadata
            
            # Try to parse the entire response as JSON
            metadata = json.loads(response_text)
            return metadata
        except json.JSONDecodeError:
            # Not valid JSON, continue with regex parsing
            pass
        
        # Look for key-value pairs in the format "Key: Value"
        kv_pairs = re.findall(r'([^:\n]+):\s*([^\n]+)', response_text)
        for key, value in kv_pairs:
            key = key.strip()
            value = value.strip()
            
            # Skip if key or value is empty
            if not key or not value:
                continue
            
            # Skip if value is too long (likely not a valid metadata value)
            if len(value) > 500:
                continue
            
            metadata[key] = value
        
        # Look for key-value pairs in the format "Key = Value"
        kv_pairs = re.findall(r'([^=\n]+)=\s*([^\n]+)', response_text)
        for key, value in kv_pairs:
            key = key.strip()
            value = value.strip()
            
            # Skip if key or value is empty
            if not key or not value:
                continue
            
            # Skip if value is too long (likely not a valid metadata value)
            if len(value) > 500:
                continue
            
            metadata[key] = value
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error parsing freeform response: {str(e)}")
        return {}

def parse_structured_response(response_text: str, expected_fields: List[str]) -> Dict[str, str]:
    """
    Parse structured response to extract JSON
    
    Args:
        response_text: AI response text
        expected_fields: List of expected field names
        
    Returns:
        dict: Extracted key-value pairs
    """
    metadata = {}
    
    try:
        # Try to parse as JSON first
        try:
            # Check if the response contains a JSON object
            json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                metadata = json.loads(json_str)
                return metadata
            
            # Try to parse the entire response as JSON
            metadata = json.loads(response_text)
            return metadata
        except json.JSONDecodeError:
            # Not valid JSON, continue with regex parsing
            pass
        
        # If JSON parsing failed, try to extract field values using regex
        for field in expected_fields:
            # Look for field in the format "field: value" or "field = value" or "field" : "value"
            field_pattern = re.escape(field)
            
            # Try different formats
            patterns = [
                rf'"{field_pattern}"\s*:\s*"([^"]+)"',  # "field": "value"
                rf'"{field_pattern}"\s*:\s*([^",\s]+)',  # "field": value
                rf'{field_pattern}\s*:\s*"([^"]+)"',     # field: "value"
                rf'{field_pattern}\s*:\s*([^,\s]+)',     # field: value
                rf'{field_pattern}\s*=\s*"([^"]+)"',     # field = "value"
                rf'{field_pattern}\s*=\s*([^\s]+)',      # field = value
                rf'{field_pattern}\s*:\s*(.+?)(?:,|\n|$)'  # field: value (any text until comma, newline or end)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text)
                if match:
                    value = match.group(1).strip()
                    
                    # Skip if value is null or empty
                    if value.lower() in ["null", "none", ""]:
                        continue
                    
                    metadata[field] = value
                    break
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error parsing structured response: {str(e)}")
        return {}

def batch_extract_metadata(file_ids: List[str], config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract metadata from multiple files in batch
    
    Args:
        file_ids: List of Box file IDs
        config: Metadata extraction configuration
        
    Returns:
        dict: Extraction results for each file
    """
    # Use enhanced client if available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        try:
            # Get extraction method and parameters
            extraction_method = config.get("extraction_method", "freeform")
            model = config.get("ai_model", "google__gemini_2_0_flash_001")
            batch_size = config.get("batch_size", 5)
            
            if extraction_method == "freeform":
                prompt = config.get("freeform_prompt", "Extract key metadata from this document.")
                return st.session_state.enhanced_client.batch_extract_metadata_freeform(
                    file_ids=file_ids,
                    prompt=prompt,
                    model=model,
                    batch_size=batch_size
                )
            elif extraction_method == "structured":
                fields = config.get("custom_fields", [])
                return st.session_state.enhanced_client.batch_extract_metadata_structured(
                    file_ids=file_ids,
                    fields=fields,
                    model=model,
                    batch_size=batch_size
                )
            elif extraction_method == "template":
                template_id = config.get("template_id", "")
                return st.session_state.enhanced_client.batch_extract_metadata_template(
                    file_ids=file_ids,
                    template_id=template_id,
                    model=model,
                    batch_size=batch_size
                )
            else:
                raise ValueError(f"Unsupported extraction method: {extraction_method}")
        except Exception as e:
            logger.error(f"Error using enhanced client for batch metadata extraction: {str(e)}")
            # Fall back to original implementation
    
    # Original implementation for backward compatibility
    results = {}
    
    # Process each file individually
    for file_id in file_ids:
        try:
            # Get extraction method and parameters
            extraction_method = config.get("extraction_method", "freeform")
            model = config.get("ai_model", "google__gemini_2_0_flash_001")
            
            if extraction_method == "freeform":
                prompt = config.get("freeform_prompt", "Extract key metadata from this document.")
                result = extract_metadata_freeform(file_id, prompt, model)
            elif extraction_method == "structured":
                fields = config.get("custom_fields", [])
                result = extract_metadata_structured(file_id, fields, model)
            elif extraction_method == "template":
                template_id = config.get("template_id", "")
                result = extract_metadata_template(file_id, template_id, model)
            else:
                raise ValueError(f"Unsupported extraction method: {extraction_method}")
            
            results[file_id] = result
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error extracting metadata for file {file_id}: {str(e)}")
            results[file_id] = {
                "error": str(e),
                "metadata": {},
                "raw_response": ""
            }
    
    return results
