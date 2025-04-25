"""
Enhanced app.py with bug fixes for Box Metadata AI application.
This version integrates all the fixed modules to address reported issues.
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

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    # Authentication state
    if not hasattr(st.session_state, "authenticated"):
        st.session_state.authenticated = False
        logger.info("Initialized authenticated in session state")
    
    # Box client
    if not hasattr(st.session_state, "client"):
        st.session_state.client = None
        logger.info("Initialized client in session state")
    
    # Enhanced client
    if not hasattr(st.session_state, "enhanced_client"):
        st.session_state.enhanced_client = None
        logger.info("Initialized enhanced_client in session state")
    
    # Current page
    if not hasattr(st.session_state, "current_page"):
        st.session_state.current_page = "Home"
        logger.info("Initialized current_page in session state")
    
    # Last activity timestamp
    if not hasattr(st.session_state, "last_activity"):
        st.session_state.last_activity = None
        logger.info("Initialized last_activity in session state")
    
    # Selected files
    if not hasattr(st.session_state, "selected_files"):
        st.session_state.selected_files = []
        logger.info("Initialized selected_files in session state")
    
    # Selected folders
    if not hasattr(st.session_state, "selected_folders"):
        st.session_state.selected_folders = []
        logger.info("Initialized selected_folders in session state")
    
    # Metadata configuration
    if not hasattr(st.session_state, "metadata_config"):
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document including title, date, author, and any other relevant information. Format the response as key-value pairs.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "ai_model": "google__gemini_2_0_flash_001",
            "batch_size": 5
        }
        logger.info("Initialized metadata_config in session state")
    
    # Extraction results
    if not hasattr(st.session_state, "extraction_results"):
        st.session_state.extraction_results = {}
        logger.info("Initialized extraction_results in session state")
    
    # Selected result IDs
    if not hasattr(st.session_state, "selected_result_ids"):
        st.session_state.selected_result_ids = []
        logger.info("Initialized selected_result_ids in session state")
    
    # Application state
    if not hasattr(st.session_state, "application_state"):
        st.session_state.application_state = {
            "is_applying": False,
            "current_file_index": 0,
            "total_files": 0,
            "applied_files": 0,
            "results": {},
            "errors": {}
        }
        logger.info("Initialized application_state in session state")
    
    # Processing state
    if not hasattr(st.session_state, "processing_state"):
        st.session_state.processing_state = {
            "is_processing": False,
            "current_file_index": 0,
            "total_files": 0,
            "processed_files": 0,
            "results": {},
            "errors": {}
        }
        logger.info("Initialized processing_state in session state")
    
    # Debug info
    if not hasattr(st.session_state, "debug_info"):
        st.session_state.debug_info = {
            "show_debug": False,
            "api_responses": {},
            "errors": {}
        }
        logger.info("Initialized debug_info in session state")
    
    # Metadata templates
    if not hasattr(st.session_state, "metadata_templates"):
        st.session_state.metadata_templates = {}
        logger.info("Initialized metadata_templates in session state")
    
    # Feedback data
    if not hasattr(st.session_state, "feedback_data"):
        st.session_state.feedback_data = {
            "ratings": {},
            "comments": {}
        }
        logger.info("Initialized feedback_data in session state")
    
    # Document categorization
    if not hasattr(st.session_state, "document_categorization"):
        st.session_state.document_categorization = {
            "is_categorized": False,
            "categorized_files": 0,
            "total_files": 0,
            "results": {},
            "errors": {},
            "processing_state": {
                "is_processing": False,
                "current_file_index": -1,
                "current_file": "",
                "current_batch": [],
                "batch_size": 5,
                "parallel_processing": False,
                "max_workers": 2
            }
        }
        logger.info("Initialized document_categorization in session state")
    
    # Document types
    if not hasattr(st.session_state, "document_types"):
        st.session_state.document_types = []
        logger.info("Initialized document_types in session state")
    
    # Initialize template cache timestamp
    if not hasattr(st.session_state, "template_cache_timestamp"):
        st.session_state.template_cache_timestamp = None
        logger.info("Initialized template_cache_timestamp in session state")
    
    # Initialize document type to template mapping
    if not hasattr(st.session_state, "document_type_to_template"):
        st.session_state.document_type_to_template = {}
        logger.info("Initialized document_type_to_template in session state")
    
    # UI preferences
    if not hasattr(st.session_state, "ui_preferences"):
        st.session_state.ui_preferences = {
            "sidebar_expanded": True,
            "show_file_details": True,
            "show_thumbnails": True,
            "compact_view": False,
            "dark_mode": False
        }
        logger.info("Initialized ui_preferences in session state")
    
    # Performance settings
    if not hasattr(st.session_state, "performance_settings"):
        st.session_state.performance_settings = {
            "enable_caching": True,
            "cache_ttl": 300,  # 5 minutes
            "parallel_processing": True,
            "max_workers": 4,
            "batch_size": 10,
            "adaptive_retries": True
        }
        logger.info("Initialized performance_settings in session state")

# Integrate enhanced modules
def integrate_enhancements():
    """
    Integrate the enhanced document types and optimized API client into the main application.
    """
    logger.info("Integrating enhanced modules and bug fixes...")
    
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

# Main app
def main():
    """Main application function"""
    # Set page config
    st.set_page_config(
        page_title="Box Metadata AI",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Integrate enhanced modules
    integrate_enhancements()
    
    # Sidebar
    with st.sidebar:
        st.title("Box Metadata AI")
        
        # Authentication
        from modules.authentication import authentication_sidebar
        authentication_sidebar()
        
        # Initialize enhanced client if authenticated and not already initialized
        if st.session_state.authenticated and st.session_state.client and not st.session_state.enhanced_client:
            initialize_enhanced_client()
            logger.info("Initialized enhanced API client")
        
        # Navigation
        if st.session_state.authenticated:
            st.subheader("Navigation")
            
            # Navigation buttons
            if st.button("Home", use_container_width=True):
                st.session_state.current_page = "Home"
                st.rerun()
            
            if st.button("File Browser", use_container_width=True):
                st.session_state.current_page = "File Browser"
                st.rerun()
            
            if st.button("Document Types", use_container_width=True):
                st.session_state.current_page = "Document Types"
                st.rerun()
            
            if st.button("Document Categorization", use_container_width=True):
                st.session_state.current_page = "Document Categorization"
                st.rerun()
            
            if st.button("Metadata Configuration", use_container_width=True):
                st.session_state.current_page = "Metadata Configuration"
                st.rerun()
            
            if st.button("Process Files", use_container_width=True):
                st.session_state.current_page = "Process Files"
                st.rerun()
            
            if st.button("Results Viewer", use_container_width=True):
                st.session_state.current_page = "Results Viewer"
                st.rerun()
            
            # Performance Settings
            with st.expander("Performance Settings", expanded=False):
                st.session_state.performance_settings["enable_caching"] = st.checkbox(
                    "Enable Caching", 
                    value=st.session_state.performance_settings.get("enable_caching", True),
                    key="enable_caching_checkbox",
                    help="Cache API responses to improve performance"
                )
                
                st.session_state.performance_settings["parallel_processing"] = st.checkbox(
                    "Enable Parallel Processing", 
                    value=st.session_state.performance_settings.get("parallel_processing", True),
                    key="parallel_processing_checkbox",
                    help="Process multiple files in parallel"
                )
                
                st.session_state.performance_settings["max_workers"] = st.slider(
                    "Max Parallel Workers",
                    min_value=1,
                    max_value=8,
                    value=st.session_state.performance_settings.get("max_workers", 4),
                    key="max_workers_slider",
                    help="Maximum number of parallel workers (higher values may improve performance but increase resource usage)",
                    disabled=not st.session_state.performance_settings["parallel_processing"]
                )
                
                st.session_state.performance_settings["batch_size"] = st.slider(
                    "Batch Size",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.performance_settings.get("batch_size", 10),
                    key="batch_size_slider",
                    help="Number of files to process in each batch"
                )
            
            # Metadata templates
            with st.expander("Metadata Templates", expanded=False):
                from modules.metadata_template_retrieval import refresh_metadata_templates
                
                if st.button("Refresh Templates", key="refresh_templates_button"):
                    if refresh_metadata_templates():
                        st.success("Templates refreshed successfully!")
                    else:
                        st.error("Failed to refresh templates. Check logs for details.")
                
                # Display available templates
                if hasattr(st.session_state, "metadata_templates") and st.session_state.metadata_templates:
                    st.write(f"Available templates: {len(st.session_state.metadata_templates)}")
                    
                    for template_id, template in st.session_state.metadata_templates.items():
                        st.write(f"- {template['displayName']}")
                else:
                    st.info("No templates available. Click 'Refresh Templates' to load templates.")
            
            # Debug mode
            with st.expander("Debug Options", expanded=False):
                st.session_state.debug_info["show_debug"] = st.checkbox(
                    "Show Debug Information",
                    value=st.session_state.debug_info["show_debug"],
                    key="show_debug_checkbox"
                )
                
                if st.button("Clear Cache", key="clear_cache_button"):
                    # Clear cache
                    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
                        st.session_state.enhanced_client.clear_cache()
                    
                    st.success("Cache cleared!")
                
                if st.button("Show Session State", key="show_session_state_button"):
                    st.json(
                        {k: v for k, v in st.session_state.items() if k not in ["client", "enhanced_client"]}
                    )
    
    # Main content
    if not st.session_state.authenticated:
        st.title("Welcome to Box Metadata AI")
        st.write("Please authenticate with Box to continue.")
        
        # Display authentication instructions
        st.write("### Authentication Instructions")
        st.write("1. Click on 'Authenticate with Box' in the sidebar")
        st.write("2. Enter your Box credentials")
        st.write("3. Grant the necessary permissions")
        
        # Display features
        st.write("### Features")
        st.write("- Extract metadata from Box files using AI")
        st.write("- Categorize documents automatically")
        st.write("- Apply metadata to files in bulk")
        st.write("- Configure custom metadata templates")
        
        # Display benefits
        st.write("### Benefits")
        st.write("- Save time by automating metadata extraction")
        st.write("- Improve search and discovery of documents")
        st.write("- Enhance document management workflows")
        st.write("- Integrate with existing Box processes")
    else:
        # Display current page
        if st.session_state.current_page == "Home":
            st.title("Box Metadata AI")
            st.write("Welcome to Box Metadata AI! This application helps you extract and apply metadata to your Box files using AI.")
            
            # Display user journey
            from modules.user_journey_guide import user_journey_guide
            user_journey_guide()
        
        elif st.session_state.current_page == "File Browser":
            from modules.file_browser import file_browser
            file_browser()
        
        elif st.session_state.current_page == "Document Types":
            from modules.document_types_manager import document_types_manager
            document_types_manager()
        
        elif st.session_state.current_page == "Document Categorization":
            # Use the fixed document categorization module
            from modules.document_categorization_fixed import document_categorization
            document_categorization()
        
        elif st.session_state.current_page == "Metadata Configuration":
            # Use the fixed metadata configuration module
            from modules.metadata_config_fixed import metadata_config
            metadata_config()
        
        elif st.session_state.current_page == "Process Files":
            st.title("Process Files")
            
            if not st.session_state.selected_files:
                st.warning("No files selected. Please select files in the File Browser first.")
                if st.button("Go to File Browser", key="go_to_file_browser_button"):
                    st.session_state.current_page = "File Browser"
                    st.rerun()
                return
            
            # Display selected files
            st.write(f"Selected files: {len(st.session_state.selected_files)}")
            
            # Display file table
            file_data = []
            for file in st.session_state.selected_files:
                file_data.append({
                    "Name": file["name"],
                    "ID": file["id"],
                    "Size": f"{file['size'] / 1024 / 1024:.2f} MB" if "size" in file else "Unknown"
                })
            
            st.table(file_data)
            
            # Process files
            st.subheader("Process Files")
            
            # Get processing configuration from metadata_config
            config = st.session_state.metadata_config
            
            # Display configuration summary
            st.write("#### Configuration Summary")
            st.write(f"Extraction Method: {config['extraction_method'].title()}")
            
            if config["extraction_method"] == "freeform":
                st.write(f"Prompt: {config['freeform_prompt'][:100]}...")
            else:
                if config["use_template"]:
                    template_id = config["template_id"]
                    if template_id in st.session_state.metadata_templates:
                        template_name = st.session_state.metadata_templates[template_id]["displayName"]
                        st.write(f"Template: {template_name}")
                    else:
                        st.write("Template: Custom (no template selected)")
                else:
                    st.write("Template: None (using custom fields)")
                    
                    if "custom_fields" in config and config["custom_fields"]:
                        st.write(f"Custom Fields: {', '.join([field['name'] for field in config['custom_fields']])}")
            
            st.write(f"AI Model: {config['ai_model']}")
            
            # Process button
            if st.button("Start Processing", key="start_processing_button", use_container_width=True):
                # Import the fixed processing module
                from modules.processing_fixed import process_files
                
                # Start processing
                process_files(st.session_state.selected_files, config)
            
            # Cancel button (only show if processing)
            if hasattr(st.session_state, "processing_state") and st.session_state.processing_state["is_processing"]:
                if st.button("Cancel Processing", key="cancel_processing_button", use_container_width=True):
                    st.session_state.processing_state["is_processing"] = False
                    st.info("Processing cancelled.")
                    st.rerun()
        
        elif st.session_state.current_page == "Results Viewer":
            from modules.results_viewer import results_viewer
            results_viewer()

# Run the app
if __name__ == "__main__":
    main()
