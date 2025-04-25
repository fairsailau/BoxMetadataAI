"""
Document categorization module for Box Metadata AI.
This module provides functionality for categorizing documents using Box AI.
COMPREHENSIVE FIX: Resolved nested expanders issue and confidence score inconsistency.
"""

import streamlit as st
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def document_categorization():
    """
    Main function for document categorization page
    """
    st.title("Document Categorization")
    
    # Check if files are selected
    if not hasattr(st.session_state, "selected_files") or not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser.")
        
        if st.button("Go to File Browser"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        
        return
    
    # Initialize document categorization state if not exists
    if not hasattr(st.session_state, "document_categorization"):
        st.session_state.document_categorization = {
            "is_categorized": False,
            "categorized_files": 0,
            "total_files": 0,
            "results": {},  # file_id -> categorization result
            "errors": {},   # file_id -> error message
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
    
    # Get document types
    from modules.document_types_manager import get_document_types_with_descriptions
    document_types = get_document_types_with_descriptions()
    
    # Check if we have document types
    if not document_types:
        st.error("No document types defined. Please define document types first.")
        
        if st.button("Go to Document Types"):
            st.session_state.current_page = "Document Types"
            st.rerun()
        
        return
    
    # Display selected files
    st.write(f"Ready to categorize {len(st.session_state.selected_files)} files using Box AI.")
    
    # Create tabs for categorization and confidence settings
    tab1, tab2 = st.tabs(["Categorization", "Confidence Settings"])
    
    with tab1:
        # AI model selection
        st.write("### AI Model Selection")
        
        model_options = [
            "google__gemini_2_0_flash_001",
            "google__gemini_2_0_pro_001",
            "anthropic__claude_3_5_sonnet_001",
            "anthropic__claude_3_haiku_001",
            "anthropic__claude_3_opus_001",
            "azure__openai__gpt_4o_mini"
        ]
        
        selected_model = st.selectbox(
            "Select AI Model for Categorization",
            options=model_options,
            index=0,
            help="Select the AI model to use for document categorization"
        )
        
        # Categorization options
        st.write("### Categorization Options")
        
        use_two_stage = st.checkbox(
            "Use two-stage categorization",
            value=True,
            help="First stage categorizes documents, second stage validates low-confidence results"
        )
        
        # Only show second-stage threshold if two-stage is enabled
        second_stage_threshold = 0.7
        if use_two_stage:
            second_stage_threshold = st.slider(
                "Confidence threshold for second-stage",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Documents with confidence below this threshold will be processed in second stage"
            )
        
        use_multi_model = st.checkbox(
            "Use multi-model consensus",
            value=False,
            help="Use multiple AI models and combine results for higher accuracy"
        )
        
        # Start/Cancel buttons
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "Start Categorization",
                use_container_width=True,
                disabled=st.session_state.document_categorization["processing_state"]["is_processing"]
            )
        
        with col2:
            cancel_button = st.button(
                "Cancel Categorization",
                use_container_width=True,
                disabled=not st.session_state.document_categorization["processing_state"]["is_processing"]
            )
        
        # Handle start button
        if start_button:
            # Reset categorization state
            st.session_state.document_categorization["is_categorized"] = False
            st.session_state.document_categorization["categorized_files"] = 0
            st.session_state.document_categorization["total_files"] = len(st.session_state.selected_files)
            st.session_state.document_categorization["results"] = {}
            st.session_state.document_categorization["errors"] = {}
            
            # Set processing state
            st.session_state.document_categorization["processing_state"]["is_processing"] = True
            st.session_state.document_categorization["processing_state"]["current_file_index"] = 0
            st.session_state.document_categorization["processing_state"]["current_file"] = ""
            
            # Apply performance settings if available
            if hasattr(st.session_state, "performance_settings"):
                st.session_state.document_categorization["processing_state"]["parallel_processing"] = st.session_state.performance_settings.get("parallel_processing", False)
                st.session_state.document_categorization["processing_state"]["max_workers"] = st.session_state.performance_settings.get("max_workers", 2)
                st.session_state.document_categorization["processing_state"]["batch_size"] = st.session_state.performance_settings.get("batch_size", 5)
            
            # Rerun to start processing
            st.rerun()
        
        # Handle cancel button
        if cancel_button:
            st.session_state.document_categorization["processing_state"]["is_processing"] = False
            st.info("Categorization cancelled.")
            st.rerun()
        
        # Process files if in processing state
        if st.session_state.document_categorization["processing_state"]["is_processing"]:
            process_files_for_categorization(document_types, selected_model, use_two_stage, second_stage_threshold, use_multi_model)
        
        # Display results if categorized
        if st.session_state.document_categorization["is_categorized"]:
            st.success(f"Categorization complete! Processed {st.session_state.document_categorization['categorized_files']} files.")
        
        # Display results if any
        if st.session_state.document_categorization["results"]:
            display_categorization_results(document_types)
    
    with tab2:
        st.write("### Confidence Configuration")
        
        # Confidence threshold configuration
        configure_confidence_thresholds()
        
        # Confidence validation examples - FIXED: No nested expanders
        st.subheader("Confidence Validation Examples")
        
        # Display confidence validation examples directly without using expanders
        display_confidence_validation_examples()

def display_categorization_results(document_types=None):
    """
    Display categorization results
    
    Args:
        document_types: List of document types with name and description
    """
    st.write("## Categorization Results")
    
    # Create tabs for table view and detailed view
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Create table data
        table_data = []
        
        for file_id, result in st.session_state.document_categorization["results"].items():
            # Get file info
            file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
            
            if file_info:
                # FIXED: Ensure consistent confidence display between views
                confidence = result.get("confidence", 0.0)
                confidence_level = "Low"
                if confidence >= 0.8:
                    confidence_level = "High"
                elif confidence >= 0.6:
                    confidence_level = "Medium"
                
                # Add to table data
                table_data.append({
                    "File Name": file_info["name"],
                    "Document Type": result.get("document_type", "Unknown"),
                    "Confidence": f"{confidence_level} ({confidence:.2f})",
                    "Status": "Needs Verification"
                })
        
        # Display table
        if table_data:
            st.table(table_data)
        else:
            st.info("No categorization results available.")
    
    with tab2:
        # Display detailed results for each file
        for file_id, result in st.session_state.document_categorization["results"].items():
            # Get file info
            file_info = next((f for f in st.session_state.selected_files if f["id"] == file_id), None)
            
            if file_info:
                st.write(f"## {file_info['name']}")
                
                # Display category and confidence
                st.write(f"**Category:** {result.get('document_type', 'Unknown')}")
                
                # FIXED: Ensure consistent confidence display between views
                confidence = result.get("confidence", 0.0)
                st.write(f"**Overall Confidence:** {confidence:.2f}")
                
                # Create confidence meter
                st.progress(confidence)
                
                # Display confidence factors
                st.write("### Confidence Factors:")
                
                # Calculate confidence factors
                ai_reported = result.get("ai_reported_confidence", 1.0)
                response_quality = result.get("response_quality", 0.8)
                evidence_strength = result.get("evidence_strength", 0.5)
                format_match = result.get("format_match", 0.5)
                
                # Display confidence factors
                st.write("#### AI Reported Confidence")
                st.progress(ai_reported)
                st.write(f"{ai_reported:.2f}")
                
                st.write("#### Response Quality")
                st.progress(response_quality)
                st.write(f"{response_quality:.2f}")
                
                st.write("#### Evidence Strength")
                st.progress(evidence_strength)
                st.write(f"{evidence_strength:.2f}")
                
                st.write("#### Format Match")
                st.progress(format_match)
                st.write(f"{format_match:.2f}")
                
                # Display reasoning
                if "reasoning" in result and result["reasoning"]:
                    with st.expander("Reasoning", expanded=False):
                        st.write(result["reasoning"])
                
                # Override category option
                st.write("### Override Category:")
                
                # Create columns for override
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Get document type names
                    document_type_names = [dt["name"] for dt in document_types] if document_types else []
                    
                    # Create dropdown for override
                    override_category = st.selectbox(
                        "Select category",
                        options=document_type_names,
                        index=document_type_names.index(result.get("document_type", "Unknown")) if result.get("document_type", "Unknown") in document_type_names else 0,
                        key=f"override_{file_id}"
                    )
                
                with col2:
                    # Create button for applying override
                    if st.button("Apply Override", key=f"apply_override_{file_id}"):
                        # Update result with override
                        result["document_type"] = override_category
                        result["is_overridden"] = True
                        
                        # Update session state
                        st.session_state.document_categorization["results"][file_id] = result
                        
                        # Show success message
                        st.success(f"Category updated to {override_category}")
                        
                        # Rerun to update UI
                        st.rerun()
                
                st.divider()

def process_files_for_categorization(document_types, model, use_two_stage=True, second_stage_threshold=0.7, use_multi_model=False):
    """
    Process files for categorization
    
    Args:
        document_types: List of document types with name and description
        model: AI model to use
        use_two_stage: Whether to use two-stage categorization
        second_stage_threshold: Confidence threshold for second stage
        use_multi_model: Whether to use multiple models for consensus
    """
    # Get processing state
    processing_state = st.session_state.document_categorization["processing_state"]
    
    # Check if processing is complete
    if processing_state["current_file_index"] >= len(st.session_state.selected_files):
        # Mark as complete
        st.session_state.document_categorization["is_categorized"] = True
        st.session_state.document_categorization["processing_state"]["is_processing"] = False
        return
    
    # Display progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check if we should use parallel processing
    if processing_state["parallel_processing"]:
        # Process in batches
        if not processing_state["current_batch"]:
            # Create next batch
            start_idx = processing_state["current_file_index"]
            end_idx = min(start_idx + processing_state["batch_size"], len(st.session_state.selected_files))
            processing_state["current_batch"] = st.session_state.selected_files[start_idx:end_idx]
            
            # Update status
            status_text.text(f"Processing batch {start_idx // processing_state['batch_size'] + 1}...")
        
        # Process batch in parallel
        process_batch_in_parallel(
            processing_state["current_batch"],
            document_types,
            model,
            use_two_stage,
            second_stage_threshold,
            use_multi_model,
            processing_state["max_workers"]
        )
        
        # Update progress
        processed_files = len(st.session_state.document_categorization["results"]) + len(st.session_state.document_categorization["errors"])
        total_files = len(st.session_state.selected_files)
        progress = processed_files / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processed {processed_files} of {total_files} files ({progress:.0%})")
        
        # Update current file index
        processing_state["current_file_index"] += len(processing_state["current_batch"])
        processing_state["current_batch"] = []
        
        # Rerun to continue processing
        time.sleep(0.1)
        st.rerun()
    else:
        # Process one file at a time
        current_file = st.session_state.selected_files[processing_state["current_file_index"]]
        processing_state["current_file"] = current_file["name"]
        
        # Update status
        status_text.text(f"Processing {current_file['name']}...")
        
        try:
            # Categorize file
            result = categorize_file(
                current_file["id"],
                document_types,
                model,
                use_two_stage,
                second_stage_threshold,
                use_multi_model
            )
            
            # Store result
            st.session_state.document_categorization["results"][current_file["id"]] = result
            st.session_state.document_categorization["categorized_files"] += 1
        except Exception as e:
            # Log error
            logger.error(f"Error categorizing file {current_file['name']}: {str(e)}")
            
            # Store error
            st.session_state.document_categorization["errors"][current_file["id"]] = str(e)
        
        # Update progress
        processing_state["current_file_index"] += 1
        progress = processing_state["current_file_index"] / len(st.session_state.selected_files)
        progress_bar.progress(progress)
        
        # Rerun to continue processing
        time.sleep(0.1)
        st.rerun()

def process_batch_in_parallel(batch, document_types, model, use_two_stage, second_stage_threshold, use_multi_model, max_workers):
    """
    Process a batch of files in parallel
    
    Args:
        batch: List of files to process
        document_types: List of document types with name and description
        model: AI model to use
        use_two_stage: Whether to use two-stage categorization
        second_stage_threshold: Confidence threshold for second stage
        use_multi_model: Whether to use multiple models for consensus
        max_workers: Maximum number of parallel workers
    """
    # Import ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    
    # Define worker function
    def worker(file):
        try:
            # Categorize file
            result = categorize_file(
                file["id"],
                document_types,
                model,
                use_two_stage,
                second_stage_threshold,
                use_multi_model
            )
            
            # Return success result
            return (file["id"], result, None)
        except Exception as e:
            # Log error
            logger.error(f"Error categorizing file {file['name']}: {str(e)}")
            
            # Return error result
            return (file["id"], None, str(e))
    
    # Process batch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = [executor.submit(worker, file) for file in batch]
        
        # Process results
        for future in futures:
            try:
                # Get result
                file_id, result, error = future.result()
                
                # Store result or error
                if error:
                    st.session_state.document_categorization["errors"][file_id] = error
                else:
                    st.session_state.document_categorization["results"][file_id] = result
                    st.session_state.document_categorization["categorized_files"] += 1
            except Exception as e:
                # Log error
                logger.error(f"Error processing future: {str(e)}")

def categorize_file(file_id, document_types, model, use_two_stage=True, second_stage_threshold=0.7, use_multi_model=False):
    """
    Categorize a single file
    
    Args:
        file_id: Box file ID
        document_types: List of document types with name and description
        model: AI model to use
        use_two_stage: Whether to use two-stage categorization
        second_stage_threshold: Confidence threshold for second stage
        use_multi_model: Whether to use multiple models for consensus
        
    Returns:
        dict: Categorization result
    """
    logger.info(f"Categorizing file: {file_id}")
    
    # Get enhanced client
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        client = st.session_state.enhanced_client
    else:
        client = st.session_state.client
    
    # Get document type names and descriptions
    document_type_names = [dt["name"] for dt in document_types]
    document_type_descriptions = {dt["name"]: dt["description"] for dt in document_types}
    
    # Create prompt
    prompt = create_categorization_prompt(document_type_names, document_type_descriptions)
    
    # First stage categorization
    first_stage_result = client.categorize_document(file_id, prompt, model)
    
    # Extract document type and confidence
    document_type = first_stage_result.get("document_type", "Unknown")
    confidence = first_stage_result.get("confidence", 0.0)
    
    # Second stage if needed
    if use_two_stage and confidence < second_stage_threshold:
        # Create second stage prompt
        second_stage_prompt = create_second_stage_prompt(document_type, document_type_descriptions.get(document_type, ""))
        
        # Second stage categorization
        second_stage_result = client.validate_document_category(file_id, second_stage_prompt, model)
        
        # Update confidence
        validation_confidence = second_stage_result.get("confidence", 0.0)
        
        # Calculate final confidence
        confidence = (confidence + validation_confidence) / 2
    
    # Multi-model consensus if needed
    if use_multi_model:
        # Define additional models
        additional_models = [
            "anthropic__claude_3_haiku_001",
            "google__gemini_2_0_pro_001"
        ]
        
        # Filter out the primary model
        additional_models = [m for m in additional_models if m != model]
        
        # Categorize with additional models
        additional_results = []
        for additional_model in additional_models:
            try:
                # Categorize with additional model
                additional_result = client.categorize_document(file_id, prompt, additional_model)
                
                # Add to results
                additional_results.append(additional_result)
            except Exception as e:
                # Log error
                logger.error(f"Error categorizing with additional model {additional_model}: {str(e)}")
        
        # Combine results if any
        if additional_results:
            # Add primary result
            all_results = [first_stage_result] + additional_results
            
            # Combine results
            combined_result = combine_categorization_results(all_results)
            
            # Update document type and confidence
            document_type = combined_result.get("document_type", document_type)
            confidence = combined_result.get("confidence", confidence)
    
    # Calculate confidence factors
    confidence_factors = calculate_confidence_factors(first_stage_result, document_type, document_type_descriptions.get(document_type, ""))
    
    # Create final result
    result = {
        "document_type": document_type,
        "confidence": confidence,
        "ai_reported_confidence": confidence_factors.get("ai_reported_confidence", 1.0),
        "response_quality": confidence_factors.get("response_quality", 0.8),
        "evidence_strength": confidence_factors.get("evidence_strength", 0.5),
        "format_match": confidence_factors.get("format_match", 0.5),
        "reasoning": first_stage_result.get("reasoning", ""),
        "raw_response": first_stage_result.get("raw_response", "")
    }
    
    return result

def create_categorization_prompt(document_types, descriptions=None):
    """
    Create prompt for document categorization
    
    Args:
        document_types: List of document type names
        descriptions: Dictionary of document type descriptions
        
    Returns:
        str: Categorization prompt
    """
    # Create base prompt
    prompt = "Analyze this document and categorize it into one of the following document types:\n\n"
    
    # Add document types with descriptions if available
    for i, doc_type in enumerate(document_types):
        prompt += f"{i+1}. {doc_type}"
        
        if descriptions and doc_type in descriptions and descriptions[doc_type]:
            prompt += f": {descriptions[doc_type]}"
        
        prompt += "\n"
    
    # Add instructions
    prompt += "\nProvide your response in the following JSON format:\n"
    prompt += "{\n"
    prompt += '  "document_type": "The most appropriate document type from the list",\n'
    prompt += '  "confidence": 0.95,\n'
    prompt += '  "reasoning": "A detailed explanation of why this document belongs to the selected category"\n'
    prompt += "}\n\n"
    
    # Add additional instructions
    prompt += "Important guidelines:\n"
    prompt += "1. Only use document types from the provided list\n"
    prompt += "2. Provide a confidence score between 0.0 and 1.0\n"
    prompt += "3. Include detailed reasoning that references specific content from the document\n"
    prompt += "4. If the document doesn't clearly fit any category, select the closest match and provide a lower confidence score\n"
    
    return prompt

def create_second_stage_prompt(document_type, description=""):
    """
    Create prompt for second stage validation
    
    Args:
        document_type: Document type to validate
        description: Description of the document type
        
    Returns:
        str: Second stage validation prompt
    """
    # Create base prompt
    prompt = f"This document has been initially categorized as '{document_type}'.\n\n"
    
    # Add description if available
    if description:
        prompt += f"Description of '{document_type}': {description}\n\n"
    
    # Add instructions
    prompt += "Please carefully analyze this document and validate whether this categorization is correct.\n\n"
    
    # Add response format
    prompt += "Provide your response in the following JSON format:\n"
    prompt += "{\n"
    prompt += '  "is_correct": true or false,\n'
    prompt += '  "confidence": 0.95,\n'
    prompt += '  "reasoning": "A detailed explanation of why this categorization is correct or incorrect"\n'
    prompt += "}\n\n"
    
    # Add additional instructions
    prompt += "Important guidelines:\n"
    prompt += "1. Focus on finding evidence that supports or contradicts the categorization\n"
    prompt += "2. Provide a confidence score between 0.0 and 1.0\n"
    prompt += "3. Include detailed reasoning that references specific content from the document\n"
    
    return prompt

def calculate_confidence_factors(result, document_type, description=""):
    """
    Calculate confidence factors for a categorization result
    
    Args:
        result: Categorization result
        document_type: Document type
        description: Description of the document type
        
    Returns:
        dict: Confidence factors
    """
    # Initialize factors
    factors = {
        "ai_reported_confidence": result.get("confidence", 1.0),
        "response_quality": 0.8,
        "evidence_strength": 0.5,
        "format_match": 0.5
    }
    
    # Calculate response quality
    raw_response = result.get("raw_response", "")
    reasoning = result.get("reasoning", "")
    
    if raw_response and reasoning:
        # Check if response is well-structured
        if "document_type" in raw_response and "confidence" in raw_response and "reasoning" in raw_response:
            factors["response_quality"] = 0.9
        
        # Check if reasoning is detailed
        if len(reasoning) > 100:
            factors["response_quality"] = min(factors["response_quality"] + 0.1, 1.0)
        elif len(reasoning) < 50:
            factors["response_quality"] = max(factors["response_quality"] - 0.2, 0.0)
    
    # Calculate evidence strength
    if reasoning:
        # Check for specific evidence
        evidence_markers = ["contains", "includes", "shows", "presents", "exhibits", "features"]
        evidence_count = sum(1 for marker in evidence_markers if marker in reasoning.lower())
        
        # Calculate evidence strength based on evidence count
        factors["evidence_strength"] = min(0.3 + (evidence_count * 0.1), 1.0)
    
    # Calculate format match
    if description and reasoning:
        # Check if reasoning mentions format characteristics from description
        description_words = set(description.lower().split())
        reasoning_words = set(reasoning.lower().split())
        
        # Calculate overlap
        overlap = len(description_words.intersection(reasoning_words))
        
        # Calculate format match based on overlap
        factors["format_match"] = min(0.3 + (overlap * 0.05), 1.0)
    
    return factors

def configure_confidence_thresholds():
    """
    Configure confidence thresholds for categorization
    """
    # Initialize confidence thresholds if not exists
    if not hasattr(st.session_state, "confidence_thresholds"):
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.8,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    # Create columns for thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Auto-accept threshold
        auto_accept = st.slider(
            "Auto-Accept Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["auto_accept"],
            step=0.05,
            help="Results with confidence above this threshold will be automatically accepted"
        )
        
        # Update threshold
        st.session_state.confidence_thresholds["auto_accept"] = auto_accept
    
    with col2:
        # Verification threshold
        verification = st.slider(
            "Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["verification"],
            step=0.05,
            help="Results with confidence above this threshold but below auto-accept will need verification"
        )
        
        # Update threshold
        st.session_state.confidence_thresholds["verification"] = verification
    
    with col3:
        # Rejection threshold
        rejection = st.slider(
            "Rejection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["rejection"],
            step=0.05,
            help="Results with confidence below this threshold will be rejected"
        )
        
        # Update threshold
        st.session_state.confidence_thresholds["rejection"] = rejection
    
    # Validate thresholds
    if auto_accept < verification:
        st.warning("Auto-accept threshold should be higher than verification threshold")
    
    if verification < rejection:
        st.warning("Verification threshold should be higher than rejection threshold")

def display_confidence_validation_examples():
    """
    Display confidence validation examples without using nested expanders
    FIXED: Completely redesigned to avoid nested expanders
    """
    # Example scenarios
    scenarios = [
        {
            "name": "Clear Invoice",
            "description": "A standard invoice with clear line items, totals, and company information",
            "confidence": 0.92,
            "category": "Invoices"
        },
        {
            "name": "Ambiguous Financial Document",
            "description": "A document with financial data that could be either an invoice or a financial report",
            "confidence": 0.65,
            "category": "Financial Report"
        },
        {
            "name": "Unusual Format Contract",
            "description": "A sales contract in an unusual format with limited standard contract language",
            "confidence": 0.45,
            "category": "Sales Contract"
        },
        {
            "name": "Unrelated Document",
            "description": "A personal letter with no clear business purpose",
            "confidence": 0.25,
            "category": "Other"
        }
    ]
    
    # Get current thresholds
    thresholds = st.session_state.confidence_thresholds
    
    # Create a grid layout for scenarios
    for i in range(0, len(scenarios), 2):
        # Create two columns
        col1, col2 = st.columns(2)
        
        # First scenario in left column
        with col1:
            if i < len(scenarios):
                scenario = scenarios[i]
                display_single_confidence_example(scenario, thresholds)
        
        # Second scenario in right column
        with col2:
            if i + 1 < len(scenarios):
                scenario = scenarios[i + 1]
                display_single_confidence_example(scenario, thresholds)

def display_single_confidence_example(scenario, thresholds):
    """
    Display a single confidence example in a card-like format
    
    Args:
        scenario: Scenario data
        thresholds: Confidence thresholds
    """
    # Determine status based on thresholds
    if scenario["confidence"] >= thresholds["auto_accept"]:
        status = "Auto-Accepted"
        color = "#28a745"
    elif scenario["confidence"] >= thresholds["verification"]:
        status = "Needs Verification"
        color = "#ffc107"
    elif scenario["confidence"] >= thresholds["rejection"]:
        status = "Low Confidence"
        color = "#ffc107"
    else:
        status = "Rejected"
        color = "#dc3545"
    
    # Create card-like container
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
        <h4>{scenario['name']} ({scenario['confidence']:.2f} confidence)</h4>
        <p><strong>Description:</strong> {scenario['description']}</p>
        <p><strong>Category:</strong> {scenario['category']}</p>
        <p><strong>Status:</strong> <span style="color: {color};">{status}</span></p>
        <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 10px;">
            <div style="width: {scenario['confidence']*100}%; background-color: {color}; height: 100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def combine_categorization_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple categorization results using weighted voting
    
    Args:
        results: List of categorization results
        
    Returns:
        dict: Combined result
    """
    if not results:
        return {
            "document_type": "Other",
            "confidence": 0.0,
            "reasoning": "No results to combine"
        }
    
    # Extract document types and confidences
    document_types = [result.get("document_type", "Unknown") for result in results]
    confidences = [result.get("confidence", 0.0) for result in results]
    
    # Count votes for each document type, weighted by confidence
    votes = {}
    for dt, conf in zip(document_types, confidences):
        if dt not in votes:
            votes[dt] = 0
        votes[dt] += conf
    
    # Find document type with highest votes
    if votes:
        document_type = max(votes.items(), key=lambda x: x[1])[0]
    else:
        document_type = "Unknown"
    
    # Calculate average confidence for selected document type
    type_confidences = [conf for dt, conf in zip(document_types, confidences) if dt == document_type]
    if type_confidences:
        confidence = sum(type_confidences) / len(type_confidences)
    else:
        confidence = 0.0
    
    # Combine reasoning
    reasoning_parts = []
    for result in results:
        if result.get("document_type", "Unknown") == document_type and "reasoning" in result:
            reasoning_parts.append(result["reasoning"])
    
    reasoning = "\n\n".join(reasoning_parts)
    
    # Create combined result
    combined_result = {
        "document_type": document_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "is_combined": True,
        "combination_method": "weighted_voting",
        "model_count": len(results)
    }
    
    return combined_result
