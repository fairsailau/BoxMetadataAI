"""
Document categorization module for Box Metadata AI.
This module provides functionality for categorizing documents using Box AI.
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
        
        # Confidence validation - FIX: Remove nested expander
        st.subheader("Confidence Validation Examples")
        validate_confidence_with_examples()

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
                # Get confidence level
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
                
                # FIX: Ensure consistent confidence display between views
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
            # Categorize document
            result = categorize_document(
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
            logger.error(f"Error categorizing document {current_file['name']}: {str(e)}")
            
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
    import concurrent.futures
    
    # Define worker function
    def worker(file):
        try:
            # Categorize document
            result = categorize_document(
                file["id"],
                document_types,
                model,
                use_two_stage,
                second_stage_threshold,
                use_multi_model
            )
            
            return file["id"], result, None
        except Exception as e:
            # Return error
            return file["id"], None, str(e)
    
    # Process batch in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(worker, file): file for file in batch}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_id, result, error = future.result()
            
            if error:
                # Store error
                st.session_state.document_categorization["errors"][file_id] = error
            else:
                # Store result
                st.session_state.document_categorization["results"][file_id] = result
                st.session_state.document_categorization["categorized_files"] += 1

def categorize_document(file_id, document_types, model, use_two_stage=True, second_stage_threshold=0.7, use_multi_model=False):
    """
    Categorize a document using Box AI
    
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
    # Check if we have a client
    if not hasattr(st.session_state, "client") or not st.session_state.client:
        raise Exception("Box client not available")
    
    # Use enhanced client if available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        client = st.session_state.enhanced_client
    else:
        # Import API client
        from modules.api_client_enhanced import BoxAPIClientEnhanced
        
        # Create API client
        client = BoxAPIClientEnhanced(st.session_state.client)
    
    # First stage categorization
    result = client.categorize_document(file_id, document_types, model)
    
    # Add additional confidence factors
    result["ai_reported_confidence"] = result.get("confidence", 0.0)
    result["response_quality"] = min(1.0, len(result.get("reasoning", "")) / 500)
    result["evidence_strength"] = 0.5  # Default value
    result["format_match"] = 0.5  # Default value
    
    # Calculate overall confidence
    confidence_weights = {
        "ai_reported_confidence": 0.5,
        "response_quality": 0.2,
        "evidence_strength": 0.15,
        "format_match": 0.15
    }
    
    overall_confidence = (
        result["ai_reported_confidence"] * confidence_weights["ai_reported_confidence"] +
        result["response_quality"] * confidence_weights["response_quality"] +
        result["evidence_strength"] * confidence_weights["evidence_strength"] +
        result["format_match"] * confidence_weights["format_match"]
    )
    
    # Update confidence
    result["confidence"] = overall_confidence
    
    # Second stage if confidence is below threshold
    if use_two_stage and overall_confidence < second_stage_threshold:
        # TODO: Implement second stage categorization
        pass
    
    # Multi-model consensus
    if use_multi_model:
        # TODO: Implement multi-model consensus
        pass
    
    return result

def configure_confidence_thresholds():
    """
    Configure confidence thresholds for categorization
    """
    # Initialize thresholds in session state if not exists
    if not hasattr(st.session_state, "confidence_thresholds"):
        st.session_state.confidence_thresholds = {
            "low": 0.4,
            "medium": 0.7,
            "high": 0.9,
            "auto_accept": 0.95,
            "auto_reject": 0.2
        }
    
    # Create columns for thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.confidence_thresholds["low"] = st.slider(
            "Low Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["low"],
            step=0.05,
            help="Confidence below this threshold is considered low"
        )
    
    with col2:
        st.session_state.confidence_thresholds["medium"] = st.slider(
            "Medium Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["medium"],
            step=0.05,
            help="Confidence below this threshold is considered medium"
        )
    
    with col3:
        st.session_state.confidence_thresholds["high"] = st.slider(
            "High Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["high"],
            step=0.05,
            help="Confidence below this threshold is considered high"
        )
    
    # Create columns for auto thresholds
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.confidence_thresholds["auto_accept"] = st.slider(
            "Auto-Accept Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["auto_accept"],
            step=0.05,
            help="Confidence above this threshold will auto-accept the categorization"
        )
    
    with col2:
        st.session_state.confidence_thresholds["auto_reject"] = st.slider(
            "Auto-Reject Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["auto_reject"],
            step=0.05,
            help="Confidence below this threshold will auto-reject the categorization"
        )

def validate_confidence_with_examples():
    """
    Validate confidence thresholds with example scenarios
    """
    # Get thresholds
    thresholds = st.session_state.confidence_thresholds if hasattr(st.session_state, "confidence_thresholds") else {
        "low": 0.4,
        "medium": 0.7,
        "high": 0.9,
        "auto_accept": 0.95,
        "auto_reject": 0.2
    }
    
    # Example scenarios
    scenarios = [
        {
            "name": "Perfect Match",
            "description": "Document perfectly matches a known type with clear structure and content",
            "category": "Invoice",
            "confidence": 0.98,
            "factors": {
                "ai_reported_confidence": 0.99,
                "response_quality": 0.95,
                "evidence_strength": 0.98,
                "format_match": 0.97
            }
        },
        {
            "name": "Strong Match",
            "description": "Document strongly matches a type with minor variations",
            "category": "Tax Form",
            "confidence": 0.85,
            "factors": {
                "ai_reported_confidence": 0.87,
                "response_quality": 0.90,
                "evidence_strength": 0.82,
                "format_match": 0.80
            }
        },
        {
            "name": "Moderate Match",
            "description": "Document has some characteristics of a type but with significant variations",
            "category": "Contract",
            "confidence": 0.65,
            "factors": {
                "ai_reported_confidence": 0.68,
                "response_quality": 0.75,
                "evidence_strength": 0.60,
                "format_match": 0.55
            }
        },
        {
            "name": "Weak Match",
            "description": "Document has few characteristics of a type and could be misclassified",
            "category": "Financial Report",
            "confidence": 0.35,
            "factors": {
                "ai_reported_confidence": 0.40,
                "response_quality": 0.50,
                "evidence_strength": 0.30,
                "format_match": 0.25
            }
        },
        {
            "name": "No Match",
            "description": "Document does not match any known type",
            "category": "Other",
            "confidence": 0.15,
            "factors": {
                "ai_reported_confidence": 0.20,
                "response_quality": 0.30,
                "evidence_strength": 0.10,
                "format_match": 0.05
            }
        }
    ]
    
    # FIX: Use columns instead of nested expanders
    for i in range(0, len(scenarios), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(scenarios):
                scenario = scenarios[i]
                st.markdown(f"**{scenario['name']} ({scenario['confidence']:.2f} confidence)**")
                st.write(f"Description: {scenario['description']}")
                st.write(f"Category: {scenario['category']}")
                st.write(f"Confidence: {scenario['confidence']:.2f}")
                
                # Determine status based on thresholds
                if scenario["confidence"] >= thresholds["auto_accept"]:
                    st.success("Status: Auto-Accept")
                elif scenario["confidence"] >= thresholds["high"]:
                    st.info("Status: High Confidence - Review")
                elif scenario["confidence"] >= thresholds["medium"]:
                    st.warning("Status: Medium Confidence - Review Carefully")
                elif scenario["confidence"] >= thresholds["low"]:
                    st.error("Status: Low Confidence - Verify")
                else:
                    st.error("Status: Auto-Reject")
                
                # Display confidence meter
                st.progress(scenario["confidence"])
        
        with col2:
            if i + 1 < len(scenarios):
                scenario = scenarios[i + 1]
                st.markdown(f"**{scenario['name']} ({scenario['confidence']:.2f} confidence)**")
                st.write(f"Description: {scenario['description']}")
                st.write(f"Category: {scenario['category']}")
                st.write(f"Confidence: {scenario['confidence']:.2f}")
                
                # Determine status based on thresholds
                if scenario["confidence"] >= thresholds["auto_accept"]:
                    st.success("Status: Auto-Accept")
                elif scenario["confidence"] >= thresholds["high"]:
                    st.info("Status: High Confidence - Review")
                elif scenario["confidence"] >= thresholds["medium"]:
                    st.warning("Status: Medium Confidence - Review Carefully")
                elif scenario["confidence"] >= thresholds["low"]:
                    st.error("Status: Low Confidence - Verify")
                else:
                    st.error("Status: Auto-Reject")
                
                # Display confidence meter
                st.progress(scenario["confidence"])
        
        st.divider()

def validate_confidence_with_examples_original():
    """
    Original function for reference - DO NOT USE (has nested expanders issue)
    """
    # Get thresholds
    thresholds = st.session_state.confidence_thresholds if hasattr(st.session_state, "confidence_thresholds") else {
        "low": 0.4,
        "medium": 0.7,
        "high": 0.9,
        "auto_accept": 0.95,
        "auto_reject": 0.2
    }
    
    # Example scenarios
    scenarios = [
        {
            "name": "Perfect Match",
            "description": "Document perfectly matches a known type with clear structure and content",
            "category": "Invoice",
            "confidence": 0.98,
            "factors": {
                "ai_reported_confidence": 0.99,
                "response_quality": 0.95,
                "evidence_strength": 0.98,
                "format_match": 0.97
            }
        },
        {
            "name": "Strong Match",
            "description": "Document strongly matches a type with minor variations",
            "category": "Tax Form",
            "confidence": 0.85,
            "factors": {
                "ai_reported_confidence": 0.87,
                "response_quality": 0.90,
                "evidence_strength": 0.82,
                "format_match": 0.80
            }
        },
        {
            "name": "Moderate Match",
            "description": "Document has some characteristics of a type but with significant variations",
            "category": "Contract",
            "confidence": 0.65,
            "factors": {
                "ai_reported_confidence": 0.68,
                "response_quality": 0.75,
                "evidence_strength": 0.60,
                "format_match": 0.55
            }
        },
        {
            "name": "Weak Match",
            "description": "Document has few characteristics of a type and could be misclassified",
            "category": "Financial Report",
            "confidence": 0.35,
            "factors": {
                "ai_reported_confidence": 0.40,
                "response_quality": 0.50,
                "evidence_strength": 0.30,
                "format_match": 0.25
            }
        },
        {
            "name": "No Match",
            "description": "Document does not match any known type",
            "category": "Other",
            "confidence": 0.15,
            "factors": {
                "ai_reported_confidence": 0.20,
                "response_quality": 0.30,
                "evidence_strength": 0.10,
                "format_match": 0.05
            }
        }
    ]
    
    # Display scenarios
    for scenario in scenarios:
        with st.expander(f"{scenario['name']} ({scenario['confidence']:.2f} confidence)", expanded=False):
            st.write(f"**Description:** {scenario['description']}")
            st.write(f"**Category:** {scenario['category']}")
            st.write(f"**Confidence:** {scenario['confidence']:.2f}")
            
            # Determine status based on thresholds
            if scenario["confidence"] >= thresholds["auto_accept"]:
                st.success("**Status:** Auto-Accept")
            elif scenario["confidence"] >= thresholds["high"]:
                st.info("**Status:** High Confidence - Review")
            elif scenario["confidence"] >= thresholds["medium"]:
                st.warning("**Status:** Medium Confidence - Review Carefully")
            elif scenario["confidence"] >= thresholds["low"]:
                st.error("**Status:** Low Confidence - Verify")
            else:
                st.error("**Status:** Auto-Reject")
            
            # Display confidence factors
            st.write("**Confidence Factors:**")
            
            for factor, value in scenario["factors"].items():
                st.write(f"- {factor.replace('_', ' ').title()}: {value:.2f}")
            
            # Display confidence meter
            st.progress(scenario["confidence"])
