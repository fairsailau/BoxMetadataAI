"""
Integration test module for testing the enhanced API client and document types manager.
This module verifies that the new implementations don't break existing functionality.
"""

import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules for testing
from modules.api_client_enhanced import BoxAPIClientEnhanced
import streamlit as st
import json

def test_api_client_enhanced_compatibility():
    """
    Test that the enhanced API client is backward compatible with the original API client.
    """
    logger.info("Testing enhanced API client compatibility")
    
    # Test that the enhanced API client has all the methods of the original API client
    from modules.api_client import BoxAPIClient
    
    # Get all public methods of the original API client
    original_methods = [method for method in dir(BoxAPIClient) if not method.startswith('_') and callable(getattr(BoxAPIClient, method))]
    
    # Get all public methods of the enhanced API client
    enhanced_methods = [method for method in dir(BoxAPIClientEnhanced) if not method.startswith('_') and callable(getattr(BoxAPIClientEnhanced, method))]
    
    # Check that all original methods are in the enhanced client
    missing_methods = [method for method in original_methods if method not in enhanced_methods]
    
    if missing_methods:
        logger.error(f"Enhanced API client is missing methods: {missing_methods}")
        return False
    
    logger.info("Enhanced API client includes all original methods")
    
    # Test that the enhanced API client can be used as a drop-in replacement
    # by checking method signatures
    for method in original_methods:
        try:
            import inspect
            original_sig = inspect.signature(getattr(BoxAPIClient, method))
            enhanced_sig = inspect.signature(getattr(BoxAPIClientEnhanced, method))
            
            # Check that all parameters in the original method are in the enhanced method
            original_params = list(original_sig.parameters.keys())
            enhanced_params = list(enhanced_sig.parameters.keys())
            
            missing_params = [param for param in original_params if param not in enhanced_params]
            
            if missing_params:
                logger.error(f"Method {method} is missing parameters: {missing_params}")
                return False
            
            logger.info(f"Method {method} has compatible signature")
        except Exception as e:
            logger.error(f"Error checking method signature for {method}: {str(e)}")
            return False
    
    logger.info("Enhanced API client is backward compatible with original API client")
    return True

def test_document_types_manager_integration():
    """
    Test that the document types manager integrates properly with the document categorization module.
    """
    logger.info("Testing document types manager integration")
    
    # Import the document types manager
    from modules.document_types_manager import get_document_type_names, get_document_types_with_descriptions
    
    # Test that the document types manager functions return expected values
    # when session state is not initialized
    if hasattr(st, 'session_state'):
        # Save current session state
        current_session_state = st.session_state
        
        # Clear session state
        delattr(st, 'session_state')
    
    # Test get_document_type_names
    default_names = get_document_type_names()
    if not isinstance(default_names, list) or "Other" not in default_names:
        logger.error(f"get_document_type_names returned unexpected value: {default_names}")
        return False
    
    # Test get_document_types_with_descriptions
    default_types = get_document_types_with_descriptions()
    if not isinstance(default_types, list) or len(default_types) < 1:
        logger.error(f"get_document_types_with_descriptions returned unexpected value: {default_types}")
        return False
    
    # Check that each type has name and description
    for dt in default_types:
        if not isinstance(dt, dict) or "name" not in dt or "description" not in dt:
            logger.error(f"Invalid document type format: {dt}")
            return False
    
    logger.info("Document types manager functions return expected values")
    
    # Restore session state if it was saved
    if 'current_session_state' in locals():
        st.session_state = current_session_state
    
    logger.info("Document types manager integration test passed")
    return True

def test_app_integration():
    """
    Test that the app.py file can be modified to include the new modules without breaking existing functionality.
    """
    logger.info("Testing app integration")
    
    # Read the app.py file
    try:
        with open(os.path.join(Path(__file__).parent.parent, "app.py"), "r") as f:
            app_content = f.read()
        
        # Check that the app.py file can be parsed
        import ast
        ast.parse(app_content)
        
        logger.info("app.py file can be parsed")
        
        # Check that the app.py file includes the new modules
        if "from modules.document_types_manager import" not in app_content:
            logger.error("app.py does not import document_types_manager")
            return False
        
        if "from modules.api_client_enhanced import" not in app_content:
            logger.error("app.py does not import api_client_enhanced")
            return False
        
        # Check that the app.py file initializes the enhanced client
        if "enhanced_client" not in app_content:
            logger.error("app.py does not initialize enhanced_client")
            return False
        
        # Check that the app.py file includes the Document Types page
        if "Document Types" not in app_content:
            logger.error("app.py does not include Document Types page")
            return False
        
        logger.info("app.py includes all required modules and functionality")
        
        logger.info("App integration test passed")
        return True
    
    except Exception as e:
        logger.error(f"Error testing app integration: {str(e)}")
        return False

def test_document_categorization_integration():
    """
    Test that the document categorization module integrates properly with the enhanced API client and document types manager.
    """
    logger.info("Testing document categorization integration")
    
    # Read the document_categorization.py file
    try:
        with open(os.path.join(Path(__file__).parent, "document_categorization.py"), "r") as f:
            doc_cat_content = f.read()
        
        # Check that the document_categorization.py file can be parsed
        import ast
        ast.parse(doc_cat_content)
        
        logger.info("document_categorization.py file can be parsed")
        
        # Check that the document_categorization.py file uses the document types manager
        if "from modules.document_types_manager import" not in doc_cat_content:
            logger.error("document_categorization.py does not import from document_types_manager")
            return False
        
        # Check that the document_categorization.py file uses the enhanced API client
        if "enhanced_client" not in doc_cat_content:
            logger.error("document_categorization.py does not use enhanced_client")
            return False
        
        # Check that the document_categorization.py file includes batch processing
        if "batch_categorize_documents" not in doc_cat_content:
            logger.error("document_categorization.py does not include batch_categorize_documents")
            return False
        
        logger.info("document_categorization.py includes all required functionality")
        
        logger.info("Document categorization integration test passed")
        return True
    
    except Exception as e:
        logger.error(f"Error testing document categorization integration: {str(e)}")
        return False

def test_metadata_extraction_integration():
    """
    Test that the metadata extraction module integrates properly with the enhanced API client.
    """
    logger.info("Testing metadata extraction integration")
    
    # Read the metadata_extraction.py file
    try:
        with open(os.path.join(Path(__file__).parent, "metadata_extraction.py"), "r") as f:
            metadata_extraction_content = f.read()
        
        # Check that the metadata_extraction.py file can be parsed
        import ast
        ast.parse(metadata_extraction_content)
        
        logger.info("metadata_extraction.py file can be parsed")
        
        # Check that the metadata_extraction.py file uses the enhanced API client
        if "enhanced_client" not in metadata_extraction_content:
            logger.error("metadata_extraction.py does not use enhanced_client")
            return False
        
        # Check that the metadata_extraction.py file includes batch processing
        if "batch_extract_metadata" not in metadata_extraction_content:
            logger.error("metadata_extraction.py does not include batch_extract_metadata")
            return False
        
        logger.info("metadata_extraction.py includes all required functionality")
        
        logger.info("Metadata extraction integration test passed")
        return True
    
    except Exception as e:
        logger.error(f"Error testing metadata extraction integration: {str(e)}")
        return False

def run_all_tests():
    """
    Run all integration tests and return the results.
    """
    results = {
        "api_client_enhanced_compatibility": test_api_client_enhanced_compatibility(),
        "document_types_manager_integration": test_document_types_manager_integration(),
        "app_integration": test_app_integration(),
        "document_categorization_integration": test_document_categorization_integration(),
        "metadata_extraction_integration": test_metadata_extraction_integration()
    }
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("All integration tests passed")
    else:
        logger.error("Some integration tests failed")
        for test, result in results.items():
            if not result:
                logger.error(f"Test {test} failed")
    
    return results

if __name__ == "__main__":
    run_all_tests()
