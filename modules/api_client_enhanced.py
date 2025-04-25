"""
Enhanced API client for Box with optimized API calls.
This module provides a centralized client for all Box API calls with optimizations
including connection pooling, request batching, and caching.
"""

import logging
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BoxAPIClientEnhanced:
    """
    Enhanced Box API client with optimized API calls
    """
    
    def __init__(self, client, cache_ttl=300, max_retries=3, pool_connections=10, pool_maxsize=20):
        """
        Initialize the enhanced API client
        
        Args:
            client: Original Box SDK client
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum number of retries for failed requests
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections to save in the pool
        """
        self.client = client
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Create a session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        # Mount the adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )
        self.session.mount("https://", adapter)
        
        # Thread-safe token management
        self.token_lock = threading.Lock()
        self.access_token = None
        self.token_expiry = 0
        
        # Initialize metrics
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "errors": 0,
            "endpoint_metrics": {}
        }
    
    def get_access_token(self):
        """
        Get a valid access token, refreshing if necessary
        
        Returns:
            str: Access token
        """
        with self.token_lock:
            # Check if token is expired or not set
            current_time = time.time()
            if not self.access_token or current_time >= self.token_expiry:
                # Extract token from client
                if hasattr(self.client, '_oauth'):
                    self.access_token = self.client._oauth.access_token
                    # Set expiry to 1 hour from now (typical Box token lifetime)
                    self.token_expiry = current_time + 3600
                elif hasattr(self.client, 'auth') and hasattr(self.client.auth, 'access_token'):
                    self.access_token = self.client.auth.access_token
                    self.token_expiry = current_time + 3600
                else:
                    raise ValueError("Could not retrieve access token from client")
            
            return self.access_token
    
    def call_api(self, endpoint, method="GET", params=None, data=None, headers=None, files=None):
        """
        Make an API call to Box
        
        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method
            params: Query parameters
            data: Request body
            headers: Additional headers
            files: Files to upload
            
        Returns:
            dict: API response
        """
        # Track metrics
        self.metrics["api_calls"] += 1
        if endpoint not in self.metrics["endpoint_metrics"]:
            self.metrics["endpoint_metrics"][endpoint] = {
                "calls": 0,
                "errors": 0,
                "avg_response_time": 0
            }
        self.metrics["endpoint_metrics"][endpoint]["calls"] += 1
        
        # Get access token
        access_token = self.get_access_token()
        
        # Set base URL
        base_url = "https://api.box.com/2.0/"
        
        # Remove leading slash if present
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        
        # Construct full URL
        url = f"{base_url}{endpoint}"
        
        # Set default headers
        default_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Merge with additional headers
        if headers:
            default_headers.update(headers)
        
        # Convert data to JSON if it's a dict
        json_data = None
        if data and isinstance(data, dict):
            json_data = data
            data = None
        
        # Make the API call
        start_time = time.time()
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=default_headers,
                files=files
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            if response.content:
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = {"content": response.content.decode("utf-8")}
            else:
                result = {}
            
            # Update metrics
            end_time = time.time()
            response_time = end_time - start_time
            current_avg = self.metrics["endpoint_metrics"][endpoint]["avg_response_time"]
            current_calls = self.metrics["endpoint_metrics"][endpoint]["calls"]
            new_avg = ((current_avg * (current_calls - 1)) + response_time) / current_calls
            self.metrics["endpoint_metrics"][endpoint]["avg_response_time"] = new_avg
            
            return result
        
        except requests.exceptions.RetryError:
            # Retry error
            self.metrics["retries"] += 1
            self.metrics["errors"] += 1
            self.metrics["endpoint_metrics"][endpoint]["errors"] += 1
            logger.error(f"Retry error for {endpoint}")
            raise Exception(f"Maximum retries exceeded for {endpoint}")
        
        except requests.exceptions.RequestException as e:
            # Other request error
            self.metrics["errors"] += 1
            self.metrics["endpoint_metrics"][endpoint]["errors"] += 1
            logger.error(f"API error for {endpoint}: {str(e)}")
            raise Exception(f"API error: {str(e)}")
    
    @lru_cache(maxsize=100)
    def get_file_info(self, file_id):
        """
        Get file information with caching
        
        Args:
            file_id: Box file ID
            
        Returns:
            dict: File information
        """
        return self.call_api(f"files/{file_id}")
    
    def categorize_document(self, file_id, document_types, model="google__gemini_2_0_flash_001"):
        """
        Categorize a document using Box AI API
        
        Args:
            file_id: Box file ID
            document_types: List of document types with name and description
            model: AI model to use
            
        Returns:
            dict: Document categorization result
        """
        # Create prompt with document types
        type_descriptions = "\n".join([f"- {dt['name']}: {dt['description']}" for dt in document_types])
        type_names = [dt['name'] for dt in document_types]
        
        prompt = (
            f"Analyze this document and categorize it into one of the following types:\n\n"
            f"{type_descriptions}\n\n"
            f"Provide your answer in the following format:\n"
            f"Category: [selected category]\n"
            f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
            f"Reasoning: [detailed explanation with specific evidence from the document]"
        )
        
        # Construct API URL for Box AI Ask
        api_url = "ai/ask"
        
        # Construct request body
        request_body = {
            "mode": "single_item_qa",
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
                    "mode": "default"
                }
            }
        }
        
        # Make API call
        response = self.call_api(api_url, method="POST", data=request_body)
        
        # Extract answer from response
        if "answer" in response:
            answer_text = response["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = self._parse_categorization_response(answer_text, type_names)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": "Other" if "Other" in type_names else type_names[-1],
            "confidence": 0.0,
            "reasoning": "Could not determine document type"
        }
    
    def batch_categorize_documents(self, file_ids, document_types, model="google__gemini_2_0_flash_001", batch_size=5):
        """
        Categorize multiple documents in batch
        
        Args:
            file_ids: List of Box file IDs
            document_types: List of document types with name and description
            model: AI model to use
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Categorization results for each file
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(file_ids), batch_size):
            batch = file_ids[i:i+batch_size]
            
            # Process each file in the batch
            for file_id in batch:
                try:
                    result = self.categorize_document(file_id, document_types, model)
                    results[file_id] = result
                except Exception as e:
                    logger.error(f"Error categorizing document {file_id}: {str(e)}")
                    results[file_id] = {
                        "document_type": "Other" if "Other" in [dt["name"] for dt in document_types] else document_types[-1]["name"],
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}"
                    }
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(file_ids):
                time.sleep(1)
        
        return results
    
    def extract_metadata_freeform(self, file_id, prompt, model="google__gemini_2_0_flash_001"):
        """
        Extract metadata from a file using freeform prompt
        
        Args:
            file_id: Box file ID
            prompt: Prompt for metadata extraction
            model: AI model to use
            
        Returns:
            dict: Extracted metadata
        """
        # Construct API URL for Box AI Ask
        api_url = "ai/ask"
        
        # Construct request body
        request_body = {
            "mode": "single_item_qa",
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
                    "mode": "default"
                }
            }
        }
        
        # Make API call
        response = self.call_api(api_url, method="POST", data=request_body)
        
        # Extract answer from response
        if "answer" in response:
            answer_text = response["answer"]
            
            # Parse the answer to extract key-value pairs
            metadata = self._parse_freeform_response(answer_text)
            
            return {
                "metadata": metadata,
                "raw_response": answer_text
            }
        
        # If no answer in response, return empty metadata
        return {
            "metadata": {},
            "raw_response": "No response from AI"
        }
    
    def extract_metadata_structured(self, file_id, fields, model="google__gemini_2_0_flash_001"):
        """
        Extract structured metadata from a file
        
        Args:
            file_id: Box file ID
            fields: List of fields to extract with name and description
            model: AI model to use
            
        Returns:
            dict: Extracted metadata
        """
        # Create prompt with fields
        fields_text = "\n".join([f"- {field['name']}: {field['description']}" for field in fields])
        
        prompt = (
            f"Extract the following metadata fields from this document:\n\n"
            f"{fields_text}\n\n"
            f"Provide your answer in JSON format with field names as keys and extracted values as values. "
            f"If a field is not found in the document, set its value to null."
        )
        
        # Construct API URL for Box AI Ask
        api_url = "ai/ask"
        
        # Construct request body
        request_body = {
            "mode": "single_item_qa",
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
                    "mode": "default"
                }
            }
        }
        
        # Make API call
        response = self.call_api(api_url, method="POST", data=request_body)
        
        # Extract answer from response
        if "answer" in response:
            answer_text = response["answer"]
            
            # Parse the answer to extract JSON
            metadata = self._parse_structured_response(answer_text, [field["name"] for field in fields])
            
            return {
                "metadata": metadata,
                "raw_response": answer_text
            }
        
        # If no answer in response, return empty metadata
        return {
            "metadata": {},
            "raw_response": "No response from AI"
        }
    
    def extract_metadata_template(self, file_id, template_id, model="google__gemini_2_0_flash_001"):
        """
        Extract metadata from a file using a metadata template
        
        Args:
            file_id: Box file ID
            template_id: Metadata template ID
            model: AI model to use
            
        Returns:
            dict: Extracted metadata
        """
        # Get template fields from session state
        import streamlit as st
        
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
        return self.extract_metadata_structured(file_id, fields, model)
    
    def batch_extract_metadata_freeform(self, file_ids, prompt, model="google__gemini_2_0_flash_001", batch_size=5):
        """
        Extract metadata from multiple files using freeform prompt
        
        Args:
            file_ids: List of Box file IDs
            prompt: Prompt for metadata extraction
            model: AI model to use
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Extraction results for each file
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(file_ids), batch_size):
            batch = file_ids[i:i+batch_size]
            
            # Process each file in the batch
            for file_id in batch:
                try:
                    result = self.extract_metadata_freeform(file_id, prompt, model)
                    results[file_id] = result
                except Exception as e:
                    logger.error(f"Error extracting metadata for file {file_id}: {str(e)}")
                    results[file_id] = {
                        "error": str(e),
                        "metadata": {},
                        "raw_response": ""
                    }
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(file_ids):
                time.sleep(1)
        
        return results
    
    def batch_extract_metadata_structured(self, file_ids, fields, model="google__gemini_2_0_flash_001", batch_size=5):
        """
        Extract structured metadata from multiple files
        
        Args:
            file_ids: List of Box file IDs
            fields: List of fields to extract with name and description
            model: AI model to use
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Extraction results for each file
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(file_ids), batch_size):
            batch = file_ids[i:i+batch_size]
            
            # Process each file in the batch
            for file_id in batch:
                try:
                    result = self.extract_metadata_structured(file_id, fields, model)
                    results[file_id] = result
                except Exception as e:
                    logger.error(f"Error extracting metadata for file {file_id}: {str(e)}")
                    results[file_id] = {
                        "error": str(e),
                        "metadata": {},
                        "raw_response": ""
                    }
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(file_ids):
                time.sleep(1)
        
        return results
    
    def batch_extract_metadata_template(self, file_ids, template_id, model="google__gemini_2_0_flash_001", batch_size=5):
        """
        Extract metadata from multiple files using a metadata template
        
        Args:
            file_ids: List of Box file IDs
            template_id: Metadata template ID
            model: AI model to use
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Extraction results for each file
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(file_ids), batch_size):
            batch = file_ids[i:i+batch_size]
            
            # Process each file in the batch
            for file_id in batch:
                try:
                    result = self.extract_metadata_template(file_id, template_id, model)
                    results[file_id] = result
                except Exception as e:
                    logger.error(f"Error extracting metadata for file {file_id}: {str(e)}")
                    results[file_id] = {
                        "error": str(e),
                        "metadata": {},
                        "raw_response": ""
                    }
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(file_ids):
                time.sleep(1)
        
        return results
    
    def apply_metadata(self, file_id, metadata, template_id=None):
        """
        Apply metadata to a file
        
        Args:
            file_id: Box file ID
            metadata: Metadata to apply
            template_id: Metadata template ID (optional)
            
        Returns:
            dict: API response
        """
        if template_id:
            # Apply template metadata
            api_url = f"files/{file_id}/metadata/enterprise/template"
            
            # Replace template with actual template ID
            api_url = api_url.replace("template", template_id)
            
            # Make API call
            return self.call_api(api_url, method="POST", data=metadata)
        else:
            # Apply custom metadata
            api_url = f"files/{file_id}/metadata/global/properties"
            
            # Make API call
            return self.call_api(api_url, method="POST", data=metadata)
    
    def batch_apply_metadata(self, file_metadata_map, template_id=None, batch_size=5):
        """
        Apply metadata to multiple files
        
        Args:
            file_metadata_map: Dictionary mapping file IDs to metadata
            template_id: Metadata template ID (optional)
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Results for each file
        """
        results = {}
        file_ids = list(file_metadata_map.keys())
        
        # Process in batches
        for i in range(0, len(file_ids), batch_size):
            batch = file_ids[i:i+batch_size]
            
            # Process each file in the batch
            for file_id in batch:
                try:
                    metadata = file_metadata_map[file_id]
                    result = self.apply_metadata(file_id, metadata, template_id)
                    results[file_id] = {
                        "success": True,
                        "response": result
                    }
                except Exception as e:
                    logger.error(f"Error applying metadata to file {file_id}: {str(e)}")
                    results[file_id] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(file_ids):
                time.sleep(1)
        
        return results
    
    def get_metrics(self):
        """
        Get API call metrics
        
        Returns:
            dict: API call metrics
        """
        return self.metrics
    
    def _parse_categorization_response(self, response_text, document_types):
        """
        Parse the AI response to extract document type, confidence score, and reasoning
        
        Args:
            response_text: The AI response text
            document_types: List of valid document types
            
        Returns:
            tuple: (document_type, confidence, reasoning)
        """
        # Default values
        document_type = document_types[-1] if document_types else "Other"
        confidence = 0.5
        reasoning = response_text
        
        try:
            import re
            
            # Try to extract category using regex
            category_match = re.search(r"Category:\s*([^\n]+)", response_text, re.IGNORECASE)
            if category_match:
                category_text = category_match.group(1).strip()
                # Find the closest matching document type
                for dt in document_types:
                    if dt.lower() in category_text.lower():
                        document_type = dt
                        break
            
            # Try to extract confidence using regex
            confidence_match = re.search(r"Confidence:\s*(0\.\d+|1\.0|1)", response_text, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                # If no explicit confidence, try to find confidence-related words
                confidence_words = {
                    "very high": 0.9,
                    "high": 0.8,
                    "good": 0.7,
                    "moderate": 0.6,
                    "medium": 0.5,
                    "low": 0.4,
                    "very low": 0.3,
                    "uncertain": 0.2
                }
                
                for word, value in confidence_words.items():
                    if word in response_text.lower():
                        confidence = value
                        break
            
            # Try to extract reasoning
            reasoning_match = re.search(r"Reasoning:\s*([^\n]+(?:\n[^\n]+)*)", response_text, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # If no document type was found in the structured response, try to find it in the full text
            if document_type == document_types[-1]:
                for dt in document_types:
                    if dt.lower() in response_text.lower():
                        document_type = dt
                        break
            
            return document_type, confidence, reasoning
        
        except Exception as e:
            logger.error(f"Error parsing categorization response: {str(e)}")
            return document_type, confidence, reasoning
    
    def _parse_freeform_response(self, response_text):
        """
        Parse freeform response to extract key-value pairs
        
        Args:
            response_text: AI response text
            
        Returns:
            dict: Extracted key-value pairs
        """
        metadata = {}
        
        try:
            import re
            import json
            
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
    
    def _parse_structured_response(self, response_text, expected_fields):
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
            import re
            import json
            
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
