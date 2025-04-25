"""
Enhanced app.py with comprehensive fixes for Box Metadata AI.
This version integrates all fixes for document categorization, metadata configuration, and processing.
"""

import streamlit as st
import logging
import os
import json
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules
from modules.authentication import authenticate
from modules.file_browser import file_browser
from modules.document_types_manager import document_types_manager
from modules.metadata_template_retrieval import metadata_template_retrieval

# Import fixed modules
from modules.document_categorization_comprehensive_fix import document_categorization
from modules.metadata_config_comprehensive_fix import metadata_config
from modules.processing_comprehensive_fix import process_files

def main():
    """
    Main function for the Box Metadata AI application
    """
    # Set page config
    st.set_page_config(
        page_title="Box Metadata AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display current page
    display_current_page()

def initialize_session_state():
    """
    Initialize session state variables
    """
    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize client
    if "client" not in st.session_state:
        st.session_state.client = None
    
    # Initialize enhanced client
    if "enhanced_client" not in st.session_state:
        st.session_state.enhanced_client = None
    
    # Initialize current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Authentication"
    
    # Initialize selected files
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    
    # Initialize metadata config
    if "metadata_config" not in st.session_state:
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document including title, date, author, and any other relevant information. Format the response as key-value pairs.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "apply_metadata": False,
            "application_method": "direct",
            "scope": "enterprise",
            "ai_model": "google__gemini_2_0_flash_001",
            "batch_size": 1
        }
    
    # Initialize performance settings
    if "performance_settings" not in st.session_state:
        st.session_state.performance_settings = {
            "parallel_processing": True,
            "max_workers": 2,
            "batch_size": 5,
            "use_caching": True,
            "cache_ttl": 3600
        }

def display_sidebar():
    """
    Display sidebar with navigation and tools
    """
    with st.sidebar:
        st.title("Box Metadata AI")
        
        # Display navigation
        st.subheader("Navigation")
        
        # Authentication
        if st.button("Authentication", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Authentication" else "secondary"):
            st.session_state.current_page = "Authentication"
            st.rerun()
        
        # File Browser (only if authenticated)
        if st.session_state.authenticated:
            if st.button("File Browser", use_container_width=True,
                        type="primary" if st.session_state.current_page == "File Browser" else "secondary"):
                st.session_state.current_page = "File Browser"
                st.rerun()
        
        # Document Types (only if authenticated)
        if st.session_state.authenticated:
            if st.button("Document Types", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Document Types" else "secondary"):
                st.session_state.current_page = "Document Types"
                st.rerun()
        
        # Document Categorization (only if authenticated and files selected)
        if st.session_state.authenticated and st.session_state.selected_files:
            if st.button("Document Categorization", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Document Categorization" else "secondary"):
                st.session_state.current_page = "Document Categorization"
                st.rerun()
        
        # Metadata Configuration (only if authenticated and files selected)
        if st.session_state.authenticated and st.session_state.selected_files:
            if st.button("Metadata Configuration", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Metadata Configuration" else "secondary"):
                st.session_state.current_page = "Metadata Configuration"
                st.rerun()
        
        # Process Files (only if authenticated and files selected)
        if st.session_state.authenticated and st.session_state.selected_files:
            if st.button("Process Files", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Process Files" else "secondary"):
                st.session_state.current_page = "Process Files"
                st.rerun()
        
        # Display tools
        st.subheader("Tools")
        
        # Refresh metadata templates (only if authenticated)
        if st.session_state.authenticated:
            if st.button("Refresh Metadata Templates", use_container_width=True):
                # Refresh metadata templates
                metadata_template_retrieval()
                
                # Show success message
                st.success("Metadata templates refreshed!")
        
        # Display selected files
        if st.session_state.selected_files:
            st.subheader("Selected Files")
            
            for file in st.session_state.selected_files:
                st.write(f"- {file['name']}")
            
            # Clear selection button
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.selected_files = []
                st.rerun()
        
        # Display app info
        st.subheader("About")
        st.write("Box Metadata AI v2.0")
        st.write("Extract and apply metadata to Box files using AI")

def display_current_page():
    """
    Display the current page based on session state
    """
    # Get current page
    current_page = st.session_state.current_page
    
    # Display page
    if current_page == "Authentication":
        authenticate()
    elif current_page == "File Browser":
        file_browser()
    elif current_page == "Document Types":
        document_types_manager()
    elif current_page == "Document Categorization":
        document_categorization()
    elif current_page == "Metadata Configuration":
        metadata_config()
    elif current_page == "Process Files":
        process_files()
    else:
        st.error(f"Unknown page: {current_page}")

if __name__ == "__main__":
    main()
