"""
Fixed processing module that ensures metadata extraction functions are properly imported
and fixes the processing files functionality.
"""

import streamlit as st
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = True

def process_files():
    """
    Process files for metadata extraction with Streamlit-compatible processing
    """
    st.title("Process Files")
    
    # Add debug information
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = []
    
    # Add metadata templates
    if "metadata_templates" not in st.session_state:
        st.session_state.metadata_templates = {}
    
    # Add feedback data
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}
    
    # Initialize extraction results if not exists
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = {}
    
    try:
        if not st.session_state.authenticated or not st.session_state.client:
            st.error("Please authenticate with Box first")
            return
        
        if not st.session_state.selected_files:
            st.warning("No files selected. Please select files in the File Browser first.")
            if st.button("Go to File Browser", key="go_to_file_browser_button"):
                st.session_state.current_page = "File Browser"
                st.rerun()
            return
        
        if "metadata_config" not in st.session_state or (
            st.session_state.metadata_config["extraction_method"] == "structured" and 
            not st.session_state.metadata_config["use_template"] and 
            not st.session_state.metadata_config["custom_fields"]
        ):
            st.warning("Metadata configuration is incomplete. Please configure metadata extraction parameters.")
            if st.button("Go to Metadata Configuration", key="go_to_metadata_config_button"):
                st.session_state.current_page = "Metadata Configuration"
                st.rerun()
            return
        
        # Initialize processing state
        if "processing_state" not in st.session_state:
            st.session_state.processing_state = {
                "is_processing": False,
                "processed_files": 0,
                "total_files": len(st.session_state.selected_files),
                "current_file_index": -1,
                "current_file": "",
                "results": {},
                "errors": {},
                "retries": {},
                "max_retries": 3,
                "retry_delay": 2,  # seconds
                "visualization_data": {}
            }
        
        # Display processing information
        st.write(f"Ready to process {len(st.session_state.selected_files)} files using the configured metadata extraction parameters.")
        
        # Enhanced batch processing controls
        with st.expander("Batch Processing Controls"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Batch size control
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.metadata_config.get("batch_size", 5),
                    key="batch_size_input"
                )
                st.session_state.metadata_config["batch_size"] = batch_size
                
                # Max retries control
                max_retries = st.number_input(
                    "Max Retries",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.processing_state.get("max_retries", 3),
                    key="max_retries_input"
                )
                st.session_state.processing_state["max_retries"] = max_retries
            
            with col2:
                # Retry delay control
                retry_delay = st.number_input(
                    "Retry Delay (seconds)",
                    min_value=1,
                    max_value=30,
                    value=st.session_state.processing_state.get("retry_delay", 2),
                    key="retry_delay_input"
                )
                st.session_state.processing_state["retry_delay"] = retry_delay
                
                # Processing mode
                processing_mode = st.selectbox(
                    "Processing Mode",
                    options=["Sequential", "Parallel"],
                    index=0,
                    key="processing_mode_input"
                )
                st.session_state.processing_state["processing_mode"] = processing_mode
        
        # Template management
        with st.expander("Metadata Template Management"):
            st.write("#### Save Current Configuration as Template")
            template_name = st.text_input("Template Name", key="template_name_input")
            
            if st.button("Save Template", key="save_template_button"):
                if template_name:
                    st.session_state.metadata_templates[template_name] = st.session_state.metadata_config.copy()
                    st.success(f"Template '{template_name}' saved successfully!")
                else:
                    st.warning("Please enter a template name")
            
            st.write("#### Load Template")
            if st.session_state.metadata_templates:
                template_options = list(st.session_state.metadata_templates.keys())
                selected_template = st.selectbox(
                    "Select Template",
                    options=template_options,
                    key="load_template_select"
                )
                
                if st.button("Load Template", key="load_template_button"):
                    st.session_state.metadata_config = st.session_state.metadata_templates[selected_template].copy()
                    st.success(f"Template '{selected_template}' loaded successfully!")
            else:
                st.info("No saved templates yet")
        
        # Display configuration summary
        with st.expander("Configuration Summary"):
            st.write("#### Extraction Method")
            st.write(f"Method: {st.session_state.metadata_config['extraction_method'].capitalize()}")
            
            if st.session_state.metadata_config["extraction_method"] == "structured":
                if st.session_state.metadata_config["use_template"]:
                    st.write(f"Using template: Template ID {st.session_state.metadata_config['template_id']}")
                else:
                    st.write(f"Using {len(st.session_state.metadata_config['custom_fields'])} custom fields")
                    for i, field in enumerate(st.session_state.metadata_config['custom_fields']):
                        st.write(f"- {field.get('display_name', field.get('name', ''))} ({field.get('type', 'string')})")
            else:
                st.write("Freeform prompt:")
                st.write(f"> {st.session_state.metadata_config['freeform_prompt']}")
            
            st.write(f"AI Model: {st.session_state.metadata_config['ai_model']}")
            st.write(f"Batch Size: {st.session_state.metadata_config['batch_size']}")
        
        # Display selected files
        with st.expander("Selected Files"):
            for file in st.session_state.selected_files:
                st.write(f"- {file['name']} (Type: {file['type']})")
        
        # Process files button
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "Start Processing",
                disabled=st.session_state.processing_state["is_processing"],
                use_container_width=True,
                key="start_processing_button"
            )
        
        with col2:
            cancel_button = st.button(
                "Cancel Processing",
                disabled=not st.session_state.processing_state["is_processing"],
                use_container_width=True,
                key="cancel_processing_button"
            )
        
        # Progress tracking
        progress_container = st.container()
        
        # Process files
        if start_button:
            # Reset processing state
            st.session_state.processing_state = {
                "is_processing": True,
                "processed_files": 0,
                "total_files": len(st.session_state.selected_files),
                "current_file_index": -1,
                "current_file": "",
                "results": {},
                "errors": {},
                "retries": {},
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "processing_mode": processing_mode,
                "visualization_data": {}
            }
            
            # Reset extraction results
            st.session_state.extraction_results = {}
            
            # Get metadata extraction functions
            extraction_functions = get_extraction_functions()
            
            # Process files with progress tracking
            process_files_with_progress(
                st.session_state.selected_files,
                extraction_functions,
                batch_size=batch_size,
                processing_mode=processing_mode
            )
        
        # Cancel processing
        if cancel_button and st.session_state.processing_state.get("is_processing", False):
            st.session_state.processing_state["is_processing"] = False
            st.warning("Processing cancelled")
        
        # Display processing progress
        if st.session_state.processing_state.get("is_processing", False):
            with progress_container:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                processed_files = st.session_state.processing_state["processed_files"]
                total_files = st.session_state.processing_state["total_files"]
                current_file = st.session_state.processing_state["current_file"]
                
                # Calculate progress
                progress = processed_files / total_files if total_files > 0 else 0
                
                # Update progress bar
                progress_bar.progress(progress)
                
                # Update status text
                if current_file:
                    status_text.text(f"Processing {current_file}... ({processed_files}/{total_files})")
                else:
                    status_text.text(f"Processed {processed_files}/{total_files} files")
        
        # Display processing results
        if "results" in st.session_state.processing_state and st.session_state.processing_state["results"]:
            st.write("### Processing Results")
            
            # Display success message
            processed_files = len(st.session_state.processing_state["results"])
            error_files = len(st.session_state.processing_state["errors"]) if "errors" in st.session_state.processing_state else 0
            
            if error_files == 0:
                st.success(f"Processing complete! Successfully processed {processed_files} files.")
            else:
                st.warning(f"Processing complete! Successfully processed {processed_files} files with {error_files} errors.")
            
            # Display errors if any
            if "errors" in st.session_state.processing_state and st.session_state.processing_state["errors"]:
                st.write("### Errors")
                
                for file_id, error in st.session_state.processing_state["errors"].items():
                    # Find file name
                    file_name = ""
                    for file in st.session_state.selected_files:
                        if file["id"] == file_id:
                            file_name = file["name"]
                            break
                    
                    st.error(f"{file_name}: {error}")
            
            # Continue button
            st.write("---")
            if st.button("Continue to View Results", key="continue_to_results_button", use_container_width=True):
                st.session_state.current_page = "View Results"
                st.rerun()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Error in process_files: {str(e)}")

# Helper function to extract structured data from API response
def extract_structured_data_from_response(response):
    """
    Extract structured data from various possible response structures
    
    Args:
        response (dict): API response
        
    Returns:
        dict: Extracted structured data (key-value pairs)
    """
    structured_data = {}
    extracted_text = ""
    
    # Log the response structure for debugging
    logger.info(f"Response structure: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")
    
    if isinstance(response, dict):
        # Check for answer field (contains structured data in JSON format)
        if "answer" in response and isinstance(response["answer"], dict):
            structured_data = response["answer"]
            logger.info(f"Found structured data in 'answer' field: {structured_data}")
            return structured_data
        
        # Check for answer field as string (JSON string)
        if "answer" in response and isinstance(response["answer"], str):
            try:
                answer_data = json.loads(response["answer"])
                if isinstance(answer_data, dict):
                    structured_data = answer_data
                    logger.info(f"Found structured data in 'answer' field (JSON string): {structured_data}")
                    return structured_data
            except json.JSONDecodeError:
                logger.warning(f"Could not parse 'answer' field as JSON: {response['answer']}")
        
        # Check for key-value pairs directly in response
        for key, value in response.items():
            if key not in ["error", "items", "response", "item_collection", "entries", "type", "id", "sequence_id"]:
                structured_data[key] = value
        
        # Check in response field
        if "response" in response and isinstance(response["response"], dict):
            response_obj = response["response"]
            if "answer" in response_obj and isinstance(response_obj["answer"], dict):
                structured_data = response_obj["answer"]
                logger.info(f"Found structured data in 'response.answer' field: {structured_data}")
                return structured_data
        
        # Check in items array
        if "items" in response and isinstance(response["items"], list) and len(response["items"]) > 0:
            item = response["items"][0]
            if isinstance(item, dict):
                if "answer" in item and isinstance(item["answer"], dict):
                    structured_data = item["answer"]
                    logger.info(f"Found structured data in 'items[0].answer' field: {structured_data}")
                    return structured_data
    
    # If we couldn't find structured data, return empty dict
    if not structured_data:
        logger.warning("Could not find structured data in response")
    
    return structured_data

def process_files_with_progress(files, extraction_functions, batch_size=5, processing_mode="Sequential"):
    """
    Process files with progress tracking
    
    Args:
        files: List of files to process
        extraction_functions: Dictionary of extraction functions
        batch_size: Number of files to process in parallel
        processing_mode: "Sequential" or "Parallel"
    """
    # Get total number of files
    total_files = len(files)
    
    # Update processing state
    st.session_state.processing_state["total_files"] = total_files
    
    # Process files based on mode
    if processing_mode == "Parallel":
        # Process files in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(
                    process_single_file, 
                    file, 
                    extraction_functions
                ): file for file in files
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file = future_to_file[future]
                
                try:
                    # Get result
                    result = future.result()
                    
                    # Update processing state
                    st.session_state.processing_state["processed_files"] = i + 1
                    st.session_state.processing_state["current_file"] = file["name"]
                    
                    # Store result
                    if "error" in result:
                        st.session_state.processing_state["errors"][file["id"]] = result["error"]
                    else:
                        st.session_state.processing_state["results"][file["id"]] = result
                        st.session_state.extraction_results[file["id"]] = result
                    
                    # Force Streamlit to update
                    st.rerun()
                
                except Exception as e:
                    # Log error
                    logger.error(f"Error processing file {file['name']}: {str(e)}")
                    
                    # Update processing state
                    st.session_state.processing_state["processed_files"] = i + 1
                    st.session_state.processing_state["current_file"] = file["name"]
                    st.session_state.processing_state["errors"][file["id"]] = str(e)
                    
                    # Force Streamlit to update
                    st.rerun()
    else:
        # Process files sequentially
        for i, file in enumerate(files):
            # Update processing state
            st.session_state.processing_state["current_file_index"] = i
            st.session_state.processing_state["current_file"] = file["name"]
            
            try:
                # Process file
                result = process_single_file(file, extraction_functions)
                
                # Store result
                if "error" in result:
                    st.session_state.processing_state["errors"][file["id"]] = result["error"]
                else:
                    st.session_state.processing_state["results"][file["id"]] = result
                    st.session_state.extraction_results[file["id"]] = result
            
            except Exception as e:
                # Log error
                logger.error(f"Error processing file {file['name']}: {str(e)}")
                
                # Store error
                st.session_state.processing_state["errors"][file["id"]] = str(e)
            
            # Update processed files count
            st.session_state.processing_state["processed_files"] = i + 1
            
            # Force Streamlit to update
            st.rerun()
    
    # Mark processing as complete
    st.session_state.processing_state["is_processing"] = False
    st.session_state.processing_state["current_file"] = ""
    
    # Force Streamlit to update
    st.rerun()

def process_single_file(file, extraction_functions):
    """
    Process a single file for metadata extraction
    
    Args:
        file: File to process
        extraction_functions: Dictionary of extraction functions
        
    Returns:
        dict: Extraction result
    """
    # Get file ID and name
    file_id = file["id"]
    file_name = file["name"]
    
    # Log processing start
    logger.info(f"Processing file: {file_name} (ID: {file_id})")
    
    # Get extraction method
    extraction_method = st.session_state.metadata_config["extraction_method"]
    
    # Get document type if available
    document_type = None
    if (
        hasattr(st.session_state, "document_categorization") and 
        st.session_state.document_categorization.get("is_categorized", False) and
        file_id in st.session_state.document_categorization["results"]
    ):
        document_type = st.session_state.document_categorization["results"][file_id]["document_type"]
    
    # Get extraction function based on method and document type
    extraction_function = None
    
    if extraction_method == "freeform":
        # Get freeform extraction function
        extraction_function = extraction_functions.get("extract_freeform")
        
        # Get prompt based on document type
        prompt = st.session_state.metadata_config["freeform_prompt"]
        if document_type and "document_type_prompts" in st.session_state.metadata_config:
            prompt = st.session_state.metadata_config["document_type_prompts"].get(
                document_type, prompt
            )
        
        # Extract metadata
        if extraction_function:
            result = extraction_function(file_id, prompt)
        else:
            return {"error": "Freeform extraction function not available"}
    
    elif extraction_method == "structured":
        # Get template ID based on document type
        template_id = None
        if document_type and hasattr(st.session_state, "document_type_to_template"):
            template_id = st.session_state.document_type_to_template.get(document_type)
        
        # If no template ID from document type, use general template ID
        if not template_id:
            template_id = st.session_state.metadata_config["template_id"]
        
        # Check if using template or custom fields
        if template_id:
            # Get template extraction function
            extraction_function = extraction_functions.get("extract_with_template")
            
            # Extract metadata
            if extraction_function:
                result = extraction_function(file_id, template_id)
            else:
                return {"error": "Template extraction function not available"}
        else:
            # Get custom fields extraction function
            extraction_function = extraction_functions.get("extract_with_custom_fields")
            
            # Extract metadata
            if extraction_function:
                result = extraction_function(
                    file_id, 
                    st.session_state.metadata_config["custom_fields"]
                )
            else:
                return {"error": "Custom fields extraction function not available"}
    
    else:
        return {"error": f"Unknown extraction method: {extraction_method}"}
    
    # Add file info to result
    result["file_id"] = file_id
    result["file_name"] = file_name
    result["extraction_method"] = extraction_method
    result["document_type"] = document_type
    
    # Log processing complete
    logger.info(f"Processed file: {file_name} (ID: {file_id})")
    
    return result

def get_extraction_functions():
    """
    Get metadata extraction functions
    
    Returns:
        dict: Dictionary of extraction functions
    """
    # Import metadata extraction functions
    from modules.metadata_extraction import (
        extract_metadata_freeform,
        extract_metadata_with_template,
        extract_metadata_with_custom_fields
    )
    
    # Return dictionary of extraction functions
    return {
        "extract_freeform": extract_metadata_freeform,
        "extract_with_template": extract_metadata_with_template,
        "extract_with_custom_fields": extract_metadata_with_custom_fields
    }
