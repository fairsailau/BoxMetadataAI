"""
Integration module for enhanced document types and optimized API client.
This module integrates the enhanced functionality into the main application.
"""

import streamlit as st
import logging
import os
import re
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_enhancements():
    """
    Integrate the enhanced document types and optimized API client into the main application.
    This function should be called from app.py to enable the enhancements.
    """
    logger.info("Integrating enhanced document types and optimized API client...")
    
    # Import enhanced modules
    try:
        # Import enhanced document types manager
        from modules.document_types_manager_enhanced import (
            document_types_manager_enhanced,
            get_document_types_with_descriptions_enhanced
        )
        
        # Import optimized API client
        from modules.api_client_enhanced_optimized import (
            BoxAPIClientEnhancedOptimized
        )
        
        # Override original functions with enhanced versions
        import modules.document_types_manager
        modules.document_types_manager.document_types_manager = document_types_manager_enhanced
        modules.document_types_manager.get_document_types_with_descriptions = get_document_types_with_descriptions_enhanced
        
        # Add missing re import to api_client_enhanced_optimized
        import modules.api_client_enhanced_optimized
        if 're' not in modules.api_client_enhanced_optimized.__dict__:
            modules.api_client_enhanced_optimized.re = re
        
        logger.info("Successfully integrated enhanced modules")
        return True
    except Exception as e:
        logger.error(f"Error integrating enhancements: {str(e)}")
        return False

def initialize_enhanced_client():
    """
    Initialize the enhanced API client with optimizations.
    This function should be called after authentication to replace the standard client.
    """
    if not hasattr(st.session_state, "client") or not st.session_state.client:
        logger.warning("Cannot initialize enhanced client: No client available")
        return False
    
    try:
        # Import optimized API client
        from modules.api_client_enhanced_optimized import BoxAPIClientEnhancedOptimized
        
        # Create enhanced client
        st.session_state.enhanced_client = BoxAPIClientEnhancedOptimized(st.session_state.client)
        logger.info("Successfully initialized enhanced API client with optimizations")
        return True
    except Exception as e:
        logger.error(f"Error initializing enhanced client: {str(e)}")
        return False
