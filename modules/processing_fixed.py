"""
Processing module for Box Metadata AI.
This module provides functionality for processing files with metadata extraction and application.
"""

import streamlit as st
import logging
import time
import json
from typing import Dict, Any, List, Optional

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

def process_files(files, config):
    """
    Process files with metadata extraction and application
    
    Args:
        files: List of files to process
        config: Processing configuration
    """
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
    
    # Check if processing is already in progress
    if processing_state["is_processing"]:
        # Display progress
        progress_bar = st.progress(processing_state["processed_files"] / processing_state["total_files"])
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
            
            # Return results
            return processing_state["results"], processing_state["errors"]
        
        # Process next file
        current_file = files[processing_state["current_file_index"]]
        
        # Update status
        status_text.text(f"Processing {current_file['name']} ({processing_state['current_file_index'] + 1} of {processing_state['total_files']})...")
        
        try:
            # Process file
            result = process_file(current_file["id"], config)
            
            # Store result
            processing_state["results"][current_file["id"]] = result
            processing_state["processed_files"] += 1
        except Exception as e:
            # Log error
            logger.error(f"Error processing file {current_file['name']}: {str(e)}")
            
            # Store error
            processing_state["errors"][current_file["id"]] = str(e)
        
        # Update progress
        processing_state["current_file_index"] += 1
        progress = processing_state["current_file_index"] / processing_state["total_files"]
        progress_bar.progress(progress)
        
        # Rerun to continue processing
        time.sleep(0.1)
        st.rerun()
    else:
        # Start processing
        processing_state["is_processing"] = True
        processing_state["current_file_index"] = 0
        processing_state["total_files"] = len(files)
        processing_state["processed_files"] = 0
        processing_state["results"] = {}
        processing_state["errors"] = {}
        processing_state["start_time"] = time.time()
        
        # Rerun to start processing
        st.rerun()

def process_file(file_id, config):
    """
    Process a single file with metadata extraction and application
    
    Args:
        file_id: Box file ID
        config: Processing configuration
        
    Returns:
        dict: Processing result
    """
    logger.info(f"Processing file: {file_id}")
    
    # Get extraction method
    extraction_method = config.get("extraction_method", "freeform")
    
    # Extract metadata based on method
    if extraction_method == "freeform":
        # Get prompt
        prompt = config.get("freeform_prompt", "Extract key metadata from this document.")
        
        # Extract metadata
        extraction_result = extract_metadata_freeform(
            file_id=file_id,
            prompt=prompt,
            model=config.get("ai_model", "google__gemini_2_0_flash_001")
        )
    elif extraction_method == "structured":
        # Get custom fields
        fields = config.get("custom_fields", [])
        
        # Extract metadata
        extraction_result = extract_metadata_structured(
            file_id=file_id,
            fields=fields,
            model=config.get("ai_model", "google__gemini_2_0_flash_001")
        )
    elif extraction_method == "template":
        # Get template ID
        template_id = config.get("template_id", "")
        
        logger.info(f"Using template-based extraction with template ID: {template_id}")
        
        # Extract metadata
        extraction_result = extract_metadata_template(
            file_id=file_id,
            template_id=template_id,
            model=config.get("ai_model", "google__gemini_2_0_flash_001")
        )
    else:
        raise ValueError(f"Invalid extraction method: {extraction_method}")
    
    # Apply metadata if configured
    if config.get("apply_metadata", False):
        # Get application method
        application_method = config.get("application_method", "direct")
        
        # Apply metadata
        if application_method == "direct":
            # Import application function
            from modules.direct_metadata_application_enhanced_fixed import apply_metadata_direct
            
            # Apply metadata
            application_result = apply_metadata_direct(
                file_id=file_id,
                metadata=extraction_result["metadata"],
                template_id=config.get("template_id", ""),
                scope=config.get("scope", "enterprise")
            )
            
            # Add application result to extraction result
            extraction_result["application_result"] = application_result
    
    return extraction_result

def batch_process_files(file_ids, config):
    """
    Process multiple files in batch
    
    Args:
        file_ids: List of Box file IDs
        config: Processing configuration
        
    Returns:
        dict: Processing results for each file
    """
    # Get extraction method
    extraction_method = config.get("extraction_method", "freeform")
    
    # Extract metadata in batch
    extraction_results = batch_extract_metadata(file_ids, config)
    
    # Apply metadata if configured
    if config.get("apply_metadata", False):
        # Get application method
        application_method = config.get("application_method", "direct")
        
        # Apply metadata for each file
        for file_id, extraction_result in extraction_results.items():
            try:
                # Apply metadata
                if application_method == "direct":
                    # Import application function
                    from modules.direct_metadata_application_enhanced_fixed import apply_metadata_direct
                    
                    # Apply metadata
                    application_result = apply_metadata_direct(
                        file_id=file_id,
                        metadata=extraction_result["metadata"],
                        template_id=config.get("template_id", ""),
                        scope=config.get("scope", "enterprise")
                    )
                    
                    # Add application result to extraction result
                    extraction_result["application_result"] = application_result
            except Exception as e:
                # Log error
                logger.error(f"Error applying metadata to file {file_id}: {str(e)}")
                
                # Add error to extraction result
                extraction_result["application_error"] = str(e)
    
    return extraction_results
