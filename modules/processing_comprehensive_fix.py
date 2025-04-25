"""
Processing module for Box Metadata AI.
This module provides functionality for processing files with metadata extraction and application.
COMPREHENSIVE FIX: Fixed metadata extraction functionality and added support for document-type specific extraction methods.
"""

import streamlit as st
import logging
import time
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import extraction functions directly
from modules.metadata_extraction import (
    extract_metadata_freeform,
    extract_metadata_structured,
    extract_metadata_template,
    batch_extract_metadata
)

def process_files():
    """
    Process files with metadata extraction and application
    """
    st.title("Process Files")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_process"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Check if metadata configuration has been performed
    if not hasattr(st.session_state, "metadata_config"):
        st.warning("Metadata configuration has not been performed. Please configure metadata extraction first.")
        if st.button("Go to Metadata Configuration", key="go_to_metadata_config_button"):
            st.session_state.current_page = "Metadata Configuration"
            st.rerun()
        return
    
    # Initialize processing state if not exists
    if not hasattr(st.session_state, "processing_state"):
        st.session_state.processing_state = {
            "is_processing": False,
            "current_file_index": 0,
            "total_files": 0,
            "processed_files": 0,
            "results": {},
            "errors": {},
            "start_time": None,
            "end_time": None
        }
    
    # Get processing state
    processing_state = st.session_state.processing_state
    
    # Display selected files
    st.subheader("Selected Files")
    
    # Create table data
    file_data = []
    for file in st.session_state.selected_files:
        file_data.append({
            "File Name": file["name"],
            "File ID": file["id"],
            "File Type": file["extension"] if "extension" in file else "Folder"
        })
    
    # Display table
    st.table(file_data)
    
    # Display metadata configuration
    st.subheader("Metadata Configuration")
    
    # Check if document categorization has been performed
    has_categorization = (
        hasattr(st.session_state, "document_categorization") and 
        st.session_state.document_categorization.get("is_categorized", False)
    )
    
    # Display extraction method
    if has_categorization and hasattr(st.session_state, "document_type_extraction_methods"):
        st.write("#### Document Type Extraction Methods")
        
        # Create table data
        method_data = []
        for doc_type, method in st.session_state.document_type_extraction_methods.items():
            # Get template for document type if using structured extraction
            template_name = "N/A"
            if method == "structured" and hasattr(st.session_state, "document_type_to_template"):
                template_id = st.session_state.document_type_to_template.get(doc_type, "")
                if template_id and hasattr(st.session_state, "metadata_templates") and template_id in st.session_state.metadata_templates:
                    template_name = st.session_state.metadata_templates[template_id]["displayName"]
            
            method_data.append({
                "Document Type": doc_type,
                "Extraction Method": method.capitalize(),
                "Template": template_name if method == "structured" else "N/A"
            })
        
        # Display table
        if method_data:
            st.table(method_data)
        else:
            st.info("No document type extraction methods configured.")
    else:
        st.write(f"**Extraction Method:** {st.session_state.metadata_config.get('extraction_method', 'freeform').capitalize()}")
        
        # Display template if using structured extraction
        if st.session_state.metadata_config.get("extraction_method", "freeform") == "structured":
            template_id = st.session_state.metadata_config.get("template_id", "")
            if template_id and hasattr(st.session_state, "metadata_templates") and template_id in st.session_state.metadata_templates:
                st.write(f"**Template:** {st.session_state.metadata_templates[template_id]['displayName']}")
            else:
                st.write("**Template:** None (Using custom fields)")
    
    # Display AI model
    st.write(f"**AI Model:** {st.session_state.metadata_config.get('ai_model', 'google__gemini_2_0_flash_001')}")
    
    # Display batch size
    st.write(f"**Batch Size:** {st.session_state.metadata_config.get('batch_size', 1)}")
    
    # Process files button
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button(
            "Start Processing",
            use_container_width=True,
            disabled=processing_state["is_processing"]
        )
    
    with col2:
        cancel_button = st.button(
            "Cancel Processing",
            use_container_width=True,
            disabled=not processing_state["is_processing"]
        )
    
    # Handle start button
    if start_button:
        # Reset processing state
        processing_state["is_processing"] = True
        processing_state["current_file_index"] = 0
        processing_state["total_files"] = len(st.session_state.selected_files)
        processing_state["processed_files"] = 0
        processing_state["results"] = {}
        processing_state["errors"] = {}
        processing_state["start_time"] = time.time()
        processing_state["end_time"] = None
        
        # Rerun to start processing
        st.rerun()
    
    # Handle cancel button
    if cancel_button:
        processing_state["is_processing"] = False
        st.info("Processing cancelled.")
        st.rerun()
    
    # Process files if in processing state
    if processing_state["is_processing"]:
        # Display progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if processing is complete
        if processing_state["current_file_index"] >= processing_state["total_files"]:
            # Mark as complete
            processing_state["is_processing"] = False
            processing_state["end_time"] = time.time()
            
            # Calculate processing time
            processing_time = processing_state["end_time"] - processing_state["start_time"]
            
            # Display completion message
            st.success(f"Processing complete! Processed {processing_state['processed_files']} files in {processing_time:.2f} seconds.")
        else:
            # Get batch size
            batch_size = st.session_state.metadata_config.get("batch_size", 1)
            
            # Process in batches
            start_idx = processing_state["current_file_index"]
            end_idx = min(start_idx + batch_size, processing_state["total_files"])
            
            # Get batch of files
            batch_files = st.session_state.selected_files[start_idx:end_idx]
            
            # Update status
            status_text.text(f"Processing batch {start_idx // batch_size + 1} of {(processing_state['total_files'] + batch_size - 1) // batch_size}...")
            
            # Process batch
            process_batch(batch_files)
            
            # Update progress
            processing_state["current_file_index"] = end_idx
            progress = end_idx / processing_state["total_files"]
            progress_bar.progress(progress)
            
            # Rerun to continue processing
            time.sleep(0.1)
            st.rerun()
    
    # Display results if any
    if processing_state["results"]:
        display_processing_results()

def process_batch(files):
    """
    Process a batch of files
    
    Args:
        files: List of files to process
    """
    # Get processing state
    processing_state = st.session_state.processing_state
    
    # Get batch size
    batch_size = st.session_state.metadata_config.get("batch_size", 1)
    
    # Check if we should use parallel processing
    if batch_size > 1:
        # Process in parallel
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit tasks
            futures = [executor.submit(process_single_file, file) for file in files]
            
            # Process results
            for future in futures:
                try:
                    # Get result
                    file_id, result, error = future.result()
                    
                    # Store result or error
                    if error:
                        processing_state["errors"][file_id] = error
                    else:
                        processing_state["results"][file_id] = result
                        processing_state["processed_files"] += 1
                except Exception as e:
                    # Log error
                    logger.error(f"Error processing future: {str(e)}")
    else:
        # Process sequentially
        for file in files:
            # Process file
            file_id, result, error = process_single_file(file)
            
            # Store result or error
            if error:
                processing_state["errors"][file_id] = error
            else:
                processing_state["results"][file_id] = result
                processing_state["processed_files"] += 1

def process_single_file(file):
    """
    Process a single file
    
    Args:
        file: File to process
        
    Returns:
        tuple: (file_id, result, error)
    """
    try:
        # Get file ID
        file_id = file["id"]
        
        # Get document type if available
        document_type = None
        if (hasattr(st.session_state, "document_categorization") and 
            st.session_state.document_categorization.get("is_categorized", False) and
            file_id in st.session_state.document_categorization["results"]):
            document_type = st.session_state.document_categorization["results"][file_id]["document_type"]
        
        # Get extraction method
        extraction_method = st.session_state.metadata_config.get("extraction_method", "freeform")
        
        # If document type is available and document type extraction methods are configured, use that
        if (document_type and 
            hasattr(st.session_state, "document_type_extraction_methods") and 
            document_type in st.session_state.document_type_extraction_methods):
            extraction_method = st.session_state.document_type_extraction_methods[document_type]
        
        # Get AI model
        ai_model = st.session_state.metadata_config.get("ai_model", "google__gemini_2_0_flash_001")
        
        # Extract metadata based on method
        if extraction_method == "freeform":
            # Get prompt
            prompt = st.session_state.metadata_config.get("freeform_prompt", "Extract key metadata from this document.")
            
            # If document type is available and document type prompts are configured, use that
            if (document_type and 
                "document_type_prompts" in st.session_state.metadata_config and 
                document_type in st.session_state.metadata_config["document_type_prompts"]):
                prompt = st.session_state.metadata_config["document_type_prompts"][document_type]
            
            # Extract metadata
            result = extract_metadata_freeform(
                file_id=file_id,
                prompt=prompt,
                model=ai_model
            )
        elif extraction_method == "structured":
            # Get template ID
            template_id = st.session_state.metadata_config.get("template_id", "")
            
            # If document type is available and document type to template mapping is configured, use that
            if (document_type and 
                hasattr(st.session_state, "document_type_to_template") and 
                document_type in st.session_state.document_type_to_template):
                template_id = st.session_state.document_type_to_template[document_type]
            
            # Check if we should use template or custom fields
            if template_id:
                # Extract metadata with template
                result = extract_metadata_template(
                    file_id=file_id,
                    template_id=template_id,
                    model=ai_model
                )
            else:
                # Get custom fields
                custom_fields = st.session_state.metadata_config.get("custom_fields", [])
                
                # If document type is available and document type custom fields are configured, use that
                if (document_type and 
                    "document_type_custom_fields" in st.session_state.metadata_config and 
                    document_type in st.session_state.metadata_config["document_type_custom_fields"]):
                    custom_fields = st.session_state.metadata_config["document_type_custom_fields"][document_type]
                
                # Extract metadata with custom fields
                result = extract_metadata_structured(
                    file_id=file_id,
                    fields=custom_fields,
                    model=ai_model
                )
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")
        
        # Apply metadata if configured
        if st.session_state.metadata_config.get("apply_metadata", False):
            # Get application method
            application_method = st.session_state.metadata_config.get("application_method", "direct")
            
            # Apply metadata
            if application_method == "direct":
                # Import application function
                from modules.direct_metadata_application_enhanced_fixed import apply_metadata_direct
                
                # Apply metadata
                application_result = apply_metadata_direct(
                    file_id=file_id,
                    metadata=result["metadata"],
                    template_id=template_id if extraction_method == "structured" and template_id else "",
                    scope=st.session_state.metadata_config.get("scope", "enterprise")
                )
                
                # Add application result to extraction result
                result["application_result"] = application_result
        
        # Return success
        return (file_id, result, None)
    except Exception as e:
        # Log error
        logger.error(f"Error processing file {file['name'] if 'name' in file else file['id']}: {str(e)}")
        
        # Return error
        return (file["id"], None, str(e))

def display_processing_results():
    """
    Display processing results
    """
    st.subheader("Processing Results")
    
    # Get processing state
    processing_state = st.session_state.processing_state
    
    # Create tabs for results and errors
    tab1, tab2 = st.tabs(["Results", "Errors"])
    
    with tab1:
        # Display results
        if processing_state["results"]:
            # Create table data
            result_data = []
            
            for file_id, result in processing_state["results"].items():
                # Get file info
                file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
                
                if file_info:
                    # Add to table data
                    result_data.append({
                        "File Name": file_info["name"],
                        "Status": "Success",
                        "Metadata Count": len(result["metadata"]) if "metadata" in result else 0
                    })
            
            # Display table
            st.table(result_data)
            
            # Display detailed results
            st.write("#### Detailed Results")
            
            for file_id, result in processing_state["results"].items():
                # Get file info
                file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
                
                if file_info:
                    with st.expander(f"{file_info['name']}", expanded=False):
                        # Display metadata
                        if "metadata" in result and result["metadata"]:
                            st.write("**Extracted Metadata:**")
                            
                            # Format metadata as table
                            metadata_data = []
                            for key, value in result["metadata"].items():
                                metadata_data.append({
                                    "Field": key,
                                    "Value": value
                                })
                            
                            # Display table
                            st.table(metadata_data)
                        else:
                            st.info("No metadata extracted.")
                        
                        # Display application result if available
                        if "application_result" in result:
                            st.write("**Metadata Application Result:**")
                            
                            # Display application result
                            st.json(result["application_result"])
        else:
            st.info("No processing results available.")
    
    with tab2:
        # Display errors
        if processing_state["errors"]:
            # Create table data
            error_data = []
            
            for file_id, error in processing_state["errors"].items():
                # Get file info
                file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
                
                if file_info:
                    # Add to table data
                    error_data.append({
                        "File Name": file_info["name"],
                        "Error": error
                    })
            
            # Display table
            st.table(error_data)
        else:
            st.info("No processing errors.")
    
    # Export results button
    if processing_state["results"]:
        if st.button("Export Results", key="export_results_button"):
            # Create export data
            export_data = {}
            
            for file_id, result in processing_state["results"].items():
                # Get file info
                file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
                
                if file_info:
                    # Add to export data
                    export_data[file_info["name"]] = {
                        "file_id": file_id,
                        "metadata": result.get("metadata", {})
                    }
            
            # Convert to JSON
            export_json = json.dumps(export_data, indent=2)
            
            # Display download link
            st.download_button(
                label="Download Results",
                data=export_json,
                file_name="metadata_extraction_results.json",
                mime="application/json"
            )
