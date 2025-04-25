import streamlit as st
import logging
import os
import sys
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config at the very top of the script
st.set_page_config(
    page_title="Box Metadata AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from modules.authentication import authenticate
from modules.file_browser import file_browser
from modules.document_types_manager import document_types_manager
from modules.document_categorization_fixed import document_categorization
from modules.metadata_config_fixed import metadata_config
from modules.processing_fixed import process_files
from modules.results_viewer import view_results
from modules.metadata_template_retrieval import get_metadata_templates

def main():
    """
    Main function for the Box Metadata AI application
    """
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
    
    # Initialize document types
    if "document_types" not in st.session_state:
        st.session_state.document_types = []
    
    # Initialize metadata config
    if "metadata_config" not in st.session_state:
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document including dates, amounts, parties involved, and any other relevant information.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "ai_model": "google__gemini_2_0_flash_001",
            "batch_size": 5
        }
    
    # Initialize document type to template mapping
    if "document_type_to_template" not in st.session_state:
        st.session_state.document_type_to_template = {}
    
    # Initialize metadata templates
    if "metadata_templates" not in st.session_state:
        st.session_state.metadata_templates = {}
        # Load templates
        try:
            from modules.metadata_template_retrieval import metadata_template_retrieval
            metadata_template_retrieval()
        except Exception as e:
            logger.error(f"Error loading metadata templates: {str(e)}")

def display_sidebar():
    """
    Display sidebar with navigation
    """
    st.sidebar.title("Box Metadata AI")
    
    # Navigation
    st.sidebar.subheader("Navigation")
    
    # Authentication
    if st.sidebar.button("Authentication", key="nav_authentication", use_container_width=True):
        st.session_state.current_page = "Authentication"
        st.rerun()
    
    # File Browser
    if st.sidebar.button("File Browser", key="nav_file_browser", use_container_width=True):
        st.session_state.current_page = "File Browser"
        st.rerun()
    
    # Document Types
    if st.sidebar.button("Document Types", key="nav_document_types", use_container_width=True):
        st.session_state.current_page = "Document Types"
        st.rerun()
    
    # Document Categorization
    if st.sidebar.button("Document Categorization", key="nav_document_categorization", use_container_width=True):
        st.session_state.current_page = "Document Categorization"
        st.rerun()
    
    # Metadata Configuration
    if st.sidebar.button("Metadata Configuration", key="nav_metadata_configuration", use_container_width=True):
        st.session_state.current_page = "Metadata Configuration"
        st.rerun()
    
    # Process Files
    if st.sidebar.button("Process Files", key="nav_process_files", use_container_width=True):
        st.session_state.current_page = "Process Files"
        st.rerun()
    
    # Tools
    st.sidebar.subheader("Tools")
    
    # Refresh Metadata Templates
    if st.sidebar.button("Refresh Metadata Templates", key="refresh_templates", use_container_width=True):
        with st.spinner("Refreshing metadata templates..."):
            try:
                from modules.metadata_template_retrieval import metadata_template_retrieval
                metadata_template_retrieval(force_refresh=True)
                st.sidebar.success("Metadata templates refreshed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error refreshing metadata templates: {str(e)}")
    
    # Selected Files
    st.sidebar.subheader("Selected Files")
    
    if st.session_state.selected_files:
        for file in st.session_state.selected_files:
            st.sidebar.write(f"- {file['name']}")
        
        if st.sidebar.button("Clear Selection", key="clear_selection", use_container_width=True):
            st.session_state.selected_files = []
            st.rerun()
    else:
        st.sidebar.write("No files selected")
    
    # About
    st.sidebar.subheader("About")
    st.sidebar.write("Box Metadata AI v2.0")
    st.sidebar.write("Extract and apply metadata to Box files using AI")

def display_current_page():
    """
    Display current page based on session state
    """
    current_page = st.session_state.current_page
    
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
    elif current_page == "View Results":
        view_results()
    else:
        st.error(f"Unknown page: {current_page}")

if __name__ == "__main__":
    main()
