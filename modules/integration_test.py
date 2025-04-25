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
        
        # Create a modified version of app.py with the new modules
        modified_content = app_content
        
        # Add import for document_types_manager
        if "from modules.document_types_manager import" not in app_content:
            import_line = "from modules.document_categorization import document_categorization"
            new_import = "from modules.document_categorization import document_categorization\nfrom modules.document_types_manager import document_types_manager"
            modified_content = modified_content.replace(import_line, new_import)
        
        # Add import for api_client_enhanced
        if "from modules.api_client_enhanced import" not in modified_content:
            # Add after the other imports
            import_section_end = "from modules.user_journey_guide import user_journey_guide, display_step_help"
            new_import = import_section_end + "\nfrom modules.api_client_enhanced import BoxAPIClientEnhanced"
            modified_content = modified_content.replace(import_section_end, new_import)
        
        # Add document_types_manager to the navigation
        if "Document Types" not in modified_content:
            # Find the navigation section
            nav_section = "if st.button(\"File Browser\", use_container_width=True, key=\"nav_file_browser\"):"
            # Add document types manager button after file browser
            new_nav = nav_section + """
            navigate_to("File Browser")
        
        if st.button("Document Types", use_container_width=True, key="nav_doc_types"):
            navigate_to("Document Types")"""
            modified_content = modified_content.replace(nav_section, new_nav)
        
        # Add document_types_manager to the page display section
        if "elif st.session_state.current_page == \"Document Types\":" not in modified_content:
            # Find the document categorization section
            doc_cat_section = "elif st.session_state.current_page == \"Document Categorization\":"
            # Add document types manager section before document categorization
            new_section = """elif st.session_state.current_page == "Document Types":
        document_types_manager()
    
    """ + doc_cat_section
            modified_content = modified_content.replace(doc_cat_section, new_section)
        
        # Write the modified app.py to a test file
        test_app_path = os.path.join(Path(__file__).parent.parent, "app_test.py")
        with open(test_app_path, "w") as f:
            f.write(modified_content)
        
        # Try to parse the modified file
        with open(test_app_path, "r") as f:
            modified_content = f.read()
        
        ast.parse(modified_content)
        
        logger.info("Modified app.py file can be parsed")
        
        # Clean up test file
        os.remove(test_app_path)
        
        logger.info("App integration test passed")
        return True
    
    except Exception as e:
        logger.error(f"Error testing app integration: {str(e)}")
        return False

def run_all_tests():
    """
    Run all integration tests and return the results.
    """
    results = {
        "api_client_enhanced_compatibility": test_api_client_enhanced_compatibility(),
        "document_types_manager_integration": test_document_types_manager_integration(),
        "app_integration": test_app_integration()
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
