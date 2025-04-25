"""
Enhanced API client for Box with optimized API calls and performance improvements.
This module provides a centralized client for all Box API calls with optimizations
including connection pooling, request batching, caching, and parallel processing.
"""

import logging
import json
import time
import requests
import gzip
import os
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCacheManager:
    """
    Enhanced cache manager with tiered caching and invalidation strategies
    """
    
    def __init__(self, cache_ttl=300, max_size=500):
        """
        Initialize cache manager
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            max_size: Maximum cache size
        """
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.max_size = max_size
        self.access_times = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Cleanup every minute
        self.cache_lock = threading.Lock()
        
        # Cache hit/miss metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def get(self, cache_type, key):
        """
        Get item from cache
        
        Args:
            cache_type: Type of cache (e.g., 'file_info', 'categorization')
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        with self.cache_lock:
            # Check if cleanup is needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup()
            
            # Get cache for this type
            type_cache = self.cache.get(cache_type, {})
            
            # Check if key exists and is not expired
            if key in type_cache:
                item, expiry = type_cache[key]
                if current_time < expiry:
                    # Update access time
                    self.access_times[(cache_type, key)] = current_time
                    self.metrics["hits"] += 1
                    return item
                else:
                    # Expired item
                    del type_cache[key]
                    if (cache_type, key) in self.access_times:
                        del self.access_times[(cache_type, key)]
            
            self.metrics["misses"] += 1
            return None
    
    def set(self, cache_type, key, value, ttl=None):
        """
        Set item in cache
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        with self.cache_lock:
            # Initialize cache for this type if not exists
            if cache_type not in self.cache:
                self.cache[cache_type] = {}
            
            # Set expiry time
            expiry = time.time() + (ttl if ttl is not None else self.cache_ttl)
            
            # Add to cache
            self.cache[cache_type][key] = (value, expiry)
            self.access_times[(cache_type, key)] = time.time()
            
            # Check if cache is full
            if sum(len(c) for c in self.cache.values()) > self.max_size:
                self._evict()
    
    def invalidate(self, cache_type=None, key=None):
        """
        Invalidate cache items
        
        Args:
            cache_type: Type of cache to invalidate (None for all)
            key: Specific key to invalidate (None for all in type)
        """
        with self.cache_lock:
            if cache_type is None:
                # Invalidate all cache
                self.cache = {}
                self.access_times = {}
            elif key is None:
                # Invalidate all items of this type
                if cache_type in self.cache:
                    # Remove access times for this type
                    keys_to_remove = [k for k in self.access_times if k[0] == cache_type]
                    for k in keys_to_remove:
                        del self.access_times[k]
                    
                    # Clear cache for this type
                    self.cache[cache_type] = {}
            else:
                # Invalidate specific item
                if cache_type in self.cache and key in self.cache[cache_type]:
                    del self.cache[cache_type][key]
                    if (cache_type, key) in self.access_times:
                        del self.access_times[(cache_type, key)]
    
    def _cleanup(self):
        """
        Clean up expired items
        """
        current_time = time.time()
        self.last_cleanup = current_time
        
        # Check each cache type
        for cache_type, type_cache in list(self.cache.items()):
            # Find expired items
            expired_keys = [
                key for key, (_, expiry) in type_cache.items()
                if current_time > expiry
            ]
            
            # Remove expired items
            for key in expired_keys:
                del type_cache[key]
                if (cache_type, key) in self.access_times:
                    del self.access_times[(cache_type, key)]
                self.metrics["evictions"] += 1
    
    def _evict(self):
        """
        Evict least recently used items
        """
        # Sort by access time
        sorted_keys = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # Evict 10% of items
        evict_count = max(1, int(len(sorted_keys) * 0.1))
        
        for i in range(evict_count):
            if i < len(sorted_keys):
                (cache_type, key), _ = sorted_keys[i]
                
                # Remove from cache
                if cache_type in self.cache and key in self.cache[cache_type]:
                    del self.cache[cache_type][key]
                
                # Remove from access times
                del self.access_times[(cache_type, key)]
                self.metrics["evictions"] += 1

def optimized_batch_processing(file_ids, process_function, batch_size=10, max_workers=4):
    """
    Process files in optimized batches with parallel processing
    
    Args:
        file_ids: List of file IDs to process
        process_function: Function to process each batch
        batch_size: Size of each batch
        max_workers: Maximum number of parallel workers
        
    Returns:
        dict: Combined results from all batches
    """
    results = {}
    
    # Create batches
    batches = [file_ids[i:i+batch_size] for i in range(0, len(file_ids), batch_size)]
    
    # Define batch processor with adaptive delay
    def process_batch(batch_index, batch):
        batch_results = {}
        
        # Calculate adaptive delay based on batch index
        # Later batches get longer delays to prevent rate limiting
        adaptive_delay = min(1 + (batch_index * 0.2), 3)
        
        # Process the batch
        batch_results = process_function(batch)
        
        # Add delay before next batch
        if batch_index < len(batches) - 1:
            time.sleep(adaptive_delay)
            
        return batch_results
    
    # Process batches with parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with batch index
        from functools import partial
        batch_processor = partial(process_batch)
        
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(batch_processor, i, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results = future.result()
            results.update(batch_results)
    
    return results

class BoxAPIClientEnhancedOptimized:
    """
    Enhanced Box API client with optimized API calls and performance improvements
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
        
        # Create optimized session with connection pooling
        self.session = self._create_optimized_session(max_retries, pool_connections, pool_maxsize)
        
        # Thread-safe token management
        self.token_lock = threading.Lock()
        self.access_token = None
        self.token_expiry = 0
        
        # Initialize enhanced cache manager
        self.cache_manager = EnhancedCacheManager(cache_ttl=cache_ttl)
        
        # Initialize metrics
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "errors": 0,
            "endpoint_metrics": {}
        }
    
    def _create_optimized_session(self, max_retries, pool_connections, pool_maxsize):
        """
        Create an optimized session with connection pooling and compression
        
        Args:
            max_retries: Maximum number of retries for failed requests
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections to save in the pool
            
        Returns:
            requests.Session: Optimized session
        """
        # Implement adaptive connection pooling
        pool_connections = min(20, os.cpu_count() * 5 if os.cpu_count() else pool_connections)
        pool_maxsize = min(100, pool_connections * 5)
        
        # Implement request compression
        def compressed_request_hook(request):
            # Add compression if request body is large
            if request.body and isinstance(request.body, bytes) and len(request.body) > 1024:
                request.headers['Content-Encoding'] = 'gzip'
                request.body = gzip.compress(request.body)
            return request
        
        # Implement response decompression
        def decompression_hook(response, *args, **kwargs):
            # Decompress response if compressed
            if response.headers.get('Content-Encoding') == 'gzip':
                response._content = gzip.decompress(response.content)
            return response
        
        # Implement adaptive retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            respect_retry_after_header=True
        )
        
        # Create optimized session
        session = requests.Session()
        session.hooks['response'] = [decompression_hook]
        session.hooks['request'] = [compressed_request_hook]
        
        # Mount adapter with optimized settings
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )
        session.mount("https://", adapter)
        
        return session
    
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
    
    def get_file_info(self, file_id):
        """
        Get file information with caching
        
        Args:
            file_id: Box file ID
            
        Returns:
            dict: File information
        """
        # Check cache first
        cached_info = self.cache_manager.get("file_info", file_id)
        if cached_info:
            self.metrics["cache_hits"] += 1
            return cached_info
        
        # Cache miss, get from API
        self.metrics["cache_misses"] += 1
        file_info = self.call_api(f"files/{file_id}")
        
        # Cache the result
        self.cache_manager.set("file_info", file_id, file_info)
        
        return file_info
    
    def categorize_document_enhanced(self, file_id, document_types, model="google__gemini_2_0_flash_001"):
        """
        Enhanced document categorization with structured document types
        
        Args:
            file_id: Box file ID
            document_types: List of document types with name and description
            model: AI model to use
            
        Returns:
            dict: Document categorization result
        """
        # Check cache first
        cache_key = f"{file_id}_{model}_{hash(json.dumps(document_types))}"
        cached_result = self.cache_manager.get("categorization", cache_key)
        if cached_result:
            self.metrics["cache_hits"] += 1
            return cached_result
        
        # Create prompt with document types
        type_descriptions = "\n".join([f"- {dt['name']}: {dt['description']}" for dt in document_types])
        type_names = [dt['name'] for dt in document_types]
        
        # Create structured document types for the API
        structured_types = []
        for dt in document_types:
            structured_types.append({
                "id": dt.get("id", dt["name"].lower().replace(" ", "_")),
                "name": dt["name"],
                "description": dt["description"],
                "examples": dt.get("examples", [])
            })
        
        # Enhanced prompt with more detailed instructions
        prompt = (
            f"Analyze this document and categorize it into one of the following types:\n\n"
            f"{type_descriptions}\n\n"
            f"For each type, consider the following characteristics:\n"
            f"- Document structure and format\n"
            f"- Key terminology and language\n"
            f"- Purpose and content\n"
            f"- Typical sections or components\n\n"
            f"Provide your answer in the following format:\n"
            f"Category: [selected category - must be one of the listed types]\n"
            f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
            f"Reasoning: [detailed explanation with specific evidence from the document]"
        )
        
        # Construct API URL for Box AI Ask
        api_url = "ai/ask"
        
        # Construct enhanced request body with document types metadata
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
            },
            "metadata": {
                "document_types": structured_types,
                "expected_output_format": {
                    "category": "string",
                    "confidence": "number",
                    "reasoning": "string"
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
            
            result = {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "raw_response": answer_text
            }
            
            # Cache the result
            self.cache_manager.set("categorization", cache_key, result)
            
            return result
        
        # If no answer in response, return default
        default_result = {
            "document_type": "Other" if "Other" in type_names else type_names[-1],
            "confidence": 0.0,
            "reasoning": "Could not determine document type",
            "raw_response": "No response from AI"
        }
        
        # Cache the default result with shorter TTL
        self.cache_manager.set("categorization", cache_key, default_result, ttl=60)
        
        return default_result
    
    def batch_categorize_documents_enhanced(self, file_ids, document_types, model="google__gemini_2_0_flash_001", batch_size=10, max_workers=4):
        """
        Enhanced batch categorization with parallel processing
        
        Args:
            file_ids: List of Box file IDs
            document_types: List of document types with name and description
            model: AI model to use
            batch_size: Size of each batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            dict: Categorization results for each file
        """
        # Define batch processor function
        def process_batch(batch):
            batch_results = {}
            for file_id in batch:
                try:
                    result = self.categorize_document_enhanced(file_id, document_types, model)
                    batch_results[file_id] = result
                except Exception as e:
                    logger.error(f"Error categorizing document {file_id}: {str(e)}")
                    batch_results[file_id] = {
                        "document_type": "Other" if "Other" in [dt["name"] for dt in document_types] else document_types[-1]["name"],
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}"
                    }
            return batch_results
        
        # Use optimized batch processing
        return optimized_batch_processing(
            file_ids=file_ids,
            process_function=process_batch,
            batch_size=batch_size,
            max_workers=max_workers
        )
    
    def _parse_categorization_response(self, answer_text, type_names):
        """
        Parse categorization response to extract document type, confidence, and reasoning
        
        Args:
            answer_text: Response text from AI
            type_names: List of valid document type names
            
        Returns:
            tuple: (document_type, confidence, reasoning)
        """
        document_type = None
        confidence = 0.0
        reasoning = ""
        
        try:
            # Try to extract category
            category_match = re.search(r'Category:\s*(.+?)(?:\n|$)', answer_text)
            if category_match:
                document_type = category_match.group(1).strip()
                
                # Validate against known types
                if document_type not in type_names:
                    # Try to find closest match
                    for type_name in type_names:
                        if type_name.lower() in document_type.lower():
                            document_type = type_name
                            break
                    
                    # If still not found, use default
                    if document_type not in type_names:
                        document_type = "Other" if "Other" in type_names else type_names[-1]
            
            # Try to extract confidence
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', answer_text)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.0
            
            # Try to extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+)', answer_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
        
        except Exception as e:
            logger.error(f"Error parsing categorization response: {str(e)}")
        
        # Default values if parsing failed
        if not document_type:
            document_type = "Other" if "Other" in type_names else type_names[-1]
        
        return document_type, confidence, reasoning
    
    # For backward compatibility
    def categorize_document(self, file_id, document_types, model="google__gemini_2_0_flash_001"):
        """
        Original categorize_document method for backward compatibility
        
        Args:
            file_id: Box file ID
            document_types: List of document types with name and description
            model: AI model to use
            
        Returns:
            dict: Document categorization result
        """
        # Use enhanced version but remove raw_response for backward compatibility
        result = self.categorize_document_enhanced(file_id, document_types, model)
        if "raw_response" in result:
            del result["raw_response"]
        return result
    
    # For backward compatibility
    def batch_categorize_documents(self, file_ids, document_types, model="google__gemini_2_0_flash_001", batch_size=5):
        """
        Original batch_categorize_documents method for backward compatibility
        
        Args:
            file_ids: List of Box file IDs
            document_types: List of document types with name and description
            model: AI model to use
            batch_size: Number of files to process in each batch
            
        Returns:
            dict: Categorization results for each file
        """
        # Use enhanced version but remove raw_response for backward compatibility
        results = self.batch_categorize_documents_enhanced(
            file_ids=file_ids,
            document_types=document_types,
            model=model,
            batch_size=batch_size,
            max_workers=2  # Limit workers for backward compatibility
        )
        
        # Remove raw_response for backward compatibility
        for file_id, result in results.items():
            if "raw_response" in result:
                del result["raw_response"]
        
        return results

# For backward compatibility
class BoxAPIClientEnhanced(BoxAPIClientEnhancedOptimized):
    """
    Original class name for backward compatibility
    """
    pass
