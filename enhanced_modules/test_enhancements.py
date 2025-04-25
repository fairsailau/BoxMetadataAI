"""
Test script for enhanced document types and optimized API client.
This script tests the new functionality added to the document types manager
and API client to ensure they work correctly and maintain backward compatibility.
"""

import streamlit as st
import logging
import json
import time
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import modules
from modules.document_types_manager_enhanced import (
    get_document_types_with_descriptions_enhanced,
    validate_document_types_enhanced
)
from modules.api_client_enhanced_optimized import (
    BoxAPIClientEnhancedOptimized,
    optimized_batch_processing,
    EnhancedCacheManager
)

def test_document_types_enhancement():
    """
    Test the enhanced document types functionality
    """
    logger.info("Testing document types enhancement...")
    
    # Test document types with examples
    test_document_types = [
        {
            "id": "invoice",
            "name": "Invoice",
            "description": "A bill for products or services",
            "examples": [
                {"id": "12345", "name": "Example Invoice.pdf"}
            ]
        },
        {
            "id": "contract",
            "name": "Contract",
            "description": "A legal agreement between parties",
            "examples": [
                {"id": "67890", "name": "Example Contract.pdf"}
            ]
        }
    ]
    
    # Test validation
    validation_result = validate_document_types_enhanced(test_document_types)
    logger.info(f"Document types validation result: {validation_result}")
    
    # Test getting document types with descriptions
    # Mock session state
    if not hasattr(st, "session_state"):
        class SessionState:
            pass
        st.session_state = SessionState()
    
    st.session_state.document_types = test_document_types
    
    # Get document types with descriptions
    document_types_with_desc = get_document_types_with_descriptions_enhanced()
    logger.info(f"Document types with descriptions: {json.dumps(document_types_with_desc)}")
    
    # Verify examples are included
    has_examples = all("examples" in dt for dt in document_types_with_desc)
    logger.info(f"Document types include examples: {has_examples}")
    
    return {
        "validation_result": validation_result,
        "document_types_with_desc": document_types_with_desc,
        "has_examples": has_examples
    }

def test_enhanced_cache_manager():
    """
    Test the enhanced cache manager
    """
    logger.info("Testing enhanced cache manager...")
    
    # Create cache manager
    cache_manager = EnhancedCacheManager(cache_ttl=10, max_size=5)
    
    # Test setting and getting items
    cache_manager.set("test_type", "key1", "value1")
    cache_manager.set("test_type", "key2", "value2")
    cache_manager.set("test_type", "key3", "value3")
    
    # Get items
    value1 = cache_manager.get("test_type", "key1")
    value2 = cache_manager.get("test_type", "key2")
    value3 = cache_manager.get("test_type", "key3")
    value4 = cache_manager.get("test_type", "key4")  # Should be None
    
    logger.info(f"Cache get results: {value1}, {value2}, {value3}, {value4}")
    
    # Test cache metrics
    logger.info(f"Cache metrics: {cache_manager.metrics}")
    
    # Test cache invalidation
    cache_manager.invalidate("test_type", "key1")
    value1_after_invalidate = cache_manager.get("test_type", "key1")
    logger.info(f"Value after invalidation: {value1_after_invalidate}")
    
    # Test cache eviction
    for i in range(10):
        cache_manager.set("test_type", f"eviction_key_{i}", f"value_{i}")
    
    # Check if eviction occurred
    logger.info(f"Cache eviction metrics: {cache_manager.metrics['evictions']}")
    
    return {
        "cache_hits": cache_manager.metrics["hits"],
        "cache_misses": cache_manager.metrics["misses"],
        "cache_evictions": cache_manager.metrics["evictions"]
    }

def test_optimized_batch_processing():
    """
    Test optimized batch processing
    """
    logger.info("Testing optimized batch processing...")
    
    # Create test data
    test_items = [f"item_{i}" for i in range(20)]
    
    # Define test processor function
    def process_batch(batch):
        result = {}
        for item in batch:
            result[item] = f"processed_{item}"
            time.sleep(0.1)  # Simulate processing time
        return result
    
    # Test with different batch sizes and worker counts
    start_time = time.time()
    result_1 = optimized_batch_processing(test_items, process_batch, batch_size=5, max_workers=2)
    time_1 = time.time() - start_time
    
    start_time = time.time()
    result_2 = optimized_batch_processing(test_items, process_batch, batch_size=10, max_workers=4)
    time_2 = time.time() - start_time
    
    logger.info(f"Batch processing times: {time_1:.2f}s vs {time_2:.2f}s")
    logger.info(f"Results count: {len(result_1)} vs {len(result_2)}")
    
    return {
        "time_1": time_1,
        "time_2": time_2,
        "results_count_1": len(result_1),
        "results_count_2": len(result_2),
        "speedup": time_1 / time_2 if time_2 > 0 else 0
    }

def test_api_client_mock():
    """
    Test API client with mock data
    """
    logger.info("Testing API client with mock data...")
    
    # Create mock client
    class MockClient:
        def __init__(self):
            self._oauth = type('obj', (object,), {'access_token': 'mock_token'})
    
    mock_client = MockClient()
    
    # Create enhanced API client
    api_client = BoxAPIClientEnhancedOptimized(mock_client)
    
    # Mock API call method to avoid actual API calls
    def mock_call_api(endpoint, method="GET", params=None, data=None, headers=None, files=None):
        if endpoint == "ai/ask":
            # Mock categorization response
            return {
                "answer": "Category: Invoice\nConfidence: 0.85\nReasoning: This document contains invoice-specific elements."
            }
        elif endpoint.startswith("files/"):
            # Mock file info response
            file_id = endpoint.split("/")[1]
            return {
                "id": file_id,
                "name": f"Test File {file_id}",
                "type": "file"
            }
        return {}
    
    api_client.call_api = mock_call_api
    
    # Test document categorization
    document_types = [
        {"id": "invoice", "name": "Invoice", "description": "A bill for products or services"},
        {"id": "contract", "name": "Contract", "description": "A legal agreement between parties"}
    ]
    
    result = api_client.categorize_document_enhanced("12345", document_types)
    logger.info(f"Categorization result: {json.dumps(result)}")
    
    # Test batch categorization
    batch_result = api_client.batch_categorize_documents_enhanced(
        ["12345", "67890"], 
        document_types,
        batch_size=2,
        max_workers=2
    )
    logger.info(f"Batch categorization result count: {len(batch_result)}")
    
    # Test file info with caching
    file_info_1 = api_client.get_file_info("12345")
    file_info_2 = api_client.get_file_info("12345")  # Should be cached
    
    logger.info(f"API client metrics: {api_client.metrics}")
    
    return {
        "categorization_result": result,
        "batch_result_count": len(batch_result),
        "cache_hits": api_client.metrics["cache_hits"],
        "api_calls": api_client.metrics["api_calls"]
    }

def run_all_tests():
    """
    Run all tests and return results
    """
    results = {}
    
    try:
        results["document_types"] = test_document_types_enhancement()
    except Exception as e:
        logger.error(f"Error in document types test: {str(e)}")
        results["document_types"] = {"error": str(e)}
    
    try:
        results["cache_manager"] = test_enhanced_cache_manager()
    except Exception as e:
        logger.error(f"Error in cache manager test: {str(e)}")
        results["cache_manager"] = {"error": str(e)}
    
    try:
        results["batch_processing"] = test_optimized_batch_processing()
    except Exception as e:
        logger.error(f"Error in batch processing test: {str(e)}")
        results["batch_processing"] = {"error": str(e)}
    
    try:
        results["api_client"] = test_api_client_mock()
    except Exception as e:
        logger.error(f"Error in API client test: {str(e)}")
        results["api_client"] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    print("\n=== RUNNING ENHANCEMENT TESTS ===\n")
    
    results = run_all_tests()
    
    print("\n=== TEST RESULTS SUMMARY ===\n")
    
    for test_name, test_results in results.items():
        print(f"## {test_name.replace('_', ' ').title()} Test")
        if "error" in test_results:
            print(f"❌ Error: {test_results['error']}")
        else:
            print("✅ Success")
            for key, value in test_results.items():
                print(f"  - {key}: {value}")
        print()
    
    # Overall success
    all_success = all("error" not in test_results for test_results in results.values())
    
    if all_success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. See details above.")
