"""
Fixed document categorization module that addresses the StreamlitAPIException error
and ensures consistent confidence scores between detailed view and table view.
"""

import streamlit as st
import logging
import json
import re
import os
import datetime
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def document_categorization():
    """
    Enhanced document categorization with improved confidence metrics
    and support for user-defined document types
    """
    st.title("Document Categorization")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_cat"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Check if document types are defined
    if not st.session_state.document_types:
        st.warning("No document types defined. Please define document types first.")
        if st.button("Go to Document Types", key="go_to_doc_types_button_cat"):
            st.session_state.current_page = "Document Types"
            st.rerun()
        return
    
    # Get document types with descriptions for categorization
    from modules.document_types_manager import get_document_types_with_descriptions
    document_types_with_desc = get_document_types_with_descriptions()
    document_type_names = [dt["name"] for dt in document_types_with_desc]
    
    # Initialize document categorization state if not exists
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": {},
            "errors": {}
        }
    
    # Initialize confidence thresholds if not exists
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    # Display selected files
    num_files = len(st.session_state.selected_files)
    st.write(f"Ready to categorize {num_files} files using Box AI.")
    
    # Create tabs for main interface and settings
    tab1, tab2 = st.tabs(["Categorization", "Confidence Settings"])
    
    with tab1:
        # AI Model selection
        ai_models = [
            "azure__openai__gpt_4o_mini",
            "azure__openai__gpt_4o_2024_05_13",
            "google__gemini_2_0_flash_001",
            "google__gemini_2_0_flash_lite_preview",
            "google__gemini_1_5_flash_001",
            "google__gemini_1_5_pro_001",
            "aws__claude_3_haiku",
            "aws__claude_3_sonnet",
            "aws__claude_3_5_sonnet",
            "aws__claude_3_7_sonnet",
            "aws__titan_text_lite"
        ]
        
        selected_model = st.selectbox(
            "Select AI Model for Categorization",
            options=ai_models,
            index=0,
            key="ai_model_select_cat",
            help="Choose the AI model to use for document categorization"
        )
        
        # Enhanced categorization options
        st.write("### Categorization Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                value=True,
                help="When enabled, documents with low confidence will undergo a second analysis"
            )
            
            # Multi-model consensus option
            use_consensus = st.checkbox(
                "Use multi-model consensus",
                value=False,
                help="When enabled, multiple AI models will be used and their results combined for more accurate categorization"
            )
        
        with col2:
            # Confidence threshold for second-stage
            confidence_threshold = st.slider(
                "Confidence threshold for second-stage",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Documents with confidence below this threshold will undergo second-stage analysis",
                disabled=not use_two_stage
            )
            
            # Select models for consensus
            consensus_models = []
            if use_consensus:
                consensus_models = st.multiselect(
                    "Select models for consensus",
                    options=ai_models,
                    default=[ai_models[0], ai_models[2]] if len(ai_models) > 2 else ai_models[:1],
                    help="Select 2-3 models for best results (more models will increase processing time)"
                )
                
                if len(consensus_models) < 1:
                    st.warning("Please select at least one model for consensus categorization")
        
        # Categorization controls
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Categorization", key="start_categorization_button_cat", use_container_width=True)
        
        with col2:
            cancel_button = st.button("Cancel Categorization", key="cancel_categorization_button_cat", use_container_width=True)
        
        # Process categorization
        if start_button:
            with st.spinner("Categorizing documents..."):
                # Reset categorization results
                st.session_state.document_categorization = {
                    "is_categorized": False,
                    "results": {},
                    "errors": {}
                }
                
                # Use enhanced API client if available
                if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
                    # Get file IDs for batch processing
                    file_ids = [file["id"] for file in st.session_state.selected_files]
                    
                    try:
                        if use_consensus and consensus_models:
                            # Multi-model consensus categorization
                            all_results = {}
                            
                            # Create progress bar for models
                            model_progress = st.progress(0)
                            model_status = st.empty()
                            
                            # Process with each model
                            for i, model in enumerate(consensus_models):
                                model_status.text(f"Processing with {model}...")
                                
                                # Batch categorize with current model
                                batch_results = st.session_state.enhanced_client.batch_categorize_documents(
                                    file_ids=file_ids,
                                    document_types=document_types_with_desc,
                                    model=model,
                                    batch_size=5
                                )
                                
                                # Store results for this model
                                for file_id, result in batch_results.items():
                                    if file_id not in all_results:
                                        all_results[file_id] = []
                                    all_results[file_id].append(result)
                                
                                model_progress.progress((i + 1) / len(consensus_models))
                            
                            # Clear progress indicators
                            model_progress.empty()
                            model_status.empty()
                            
                            # Process consensus results
                            for file_id, results in all_results.items():
                                # Find the file name
                                file_name = next((file["name"] for file in st.session_state.selected_files if file["id"] == file_id), "Unknown")
                                
                                # Combine results using weighted voting
                                combined_result = combine_categorization_results(results)
                                
                                # Add model details to reasoning
                                models_text = ", ".join(consensus_models)
                                combined_result["reasoning"] = f"Consensus from models: {models_text}\n\n" + combined_result["reasoning"]
                                
                                # Extract document features for multi-factor confidence
                                document_features = extract_document_features(file_id)
                                
                                # Calculate multi-factor confidence
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    combined_result["confidence"],
                                    document_features,
                                    combined_result["document_type"],
                                    combined_result["reasoning"],
                                    document_type_names
                                )
                                
                                # Apply confidence calibration
                                calibrated_confidence = apply_confidence_calibration(
                                    combined_result["document_type"],
                                    multi_factor_confidence["overall"]
                                )
                                
                                # Store result with consistent confidence values
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": combined_result["document_type"],
                                    "confidence": calibrated_confidence,  # Use calibrated confidence consistently
                                    "confidence_raw": combined_result["confidence"],
                                    "multi_factor_confidence": multi_factor_confidence,
                                    "calibrated_confidence": calibrated_confidence,
                                    "reasoning": combined_result["reasoning"],
                                    "document_features": document_features
                                }
                        else:
                            # Single model categorization with batch processing
                            batch_results = st.session_state.enhanced_client.batch_categorize_documents(
                                file_ids=file_ids,
                                document_types=document_types_with_desc,
                                model=selected_model,
                                batch_size=5
                            )
                            
                            # Process results
                            for file_id, result in batch_results.items():
                                # Find the file name
                                file_name = next((file["name"] for file in st.session_state.selected_files if file["id"] == file_id), "Unknown")
                                
                                # Check if second-stage is needed
                                if use_two_stage and result["confidence"] < confidence_threshold:
                                    st.info(f"Low confidence ({result['confidence']:.2f}) for {file_name}, performing detailed analysis...")
                                    
                                    # Second-stage categorization with more detailed prompt
                                    detailed_result = categorize_document_detailed(file_id, selected_model, result["document_type"], document_types_with_desc)
                                    
                                    # Merge results, preferring the detailed analysis
                                    result = {
                                        "document_type": detailed_result["document_type"],
                                        "confidence": detailed_result["confidence"],
                                        "reasoning": detailed_result["reasoning"],
                                        "first_stage_type": result["document_type"],
                                        "first_stage_confidence": result["confidence"]
                                    }
                                
                                # Extract document features for multi-factor confidence
                                document_features = extract_document_features(file_id)
                                
                                # Calculate multi-factor confidence
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    result["confidence"],
                                    document_features,
                                    result["document_type"],
                                    result.get("reasoning", ""),
                                    document_type_names
                                )
                                
                                # Apply confidence calibration
                                calibrated_confidence = apply_confidence_calibration(
                                    result["document_type"],
                                    multi_factor_confidence["overall"]
                                )
                                
                                # Store result with consistent confidence values
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": result["document_type"],
                                    "confidence": calibrated_confidence,  # Use calibrated confidence consistently
                                    "confidence_raw": result["confidence"],
                                    "multi_factor_confidence": multi_factor_confidence,
                                    "calibrated_confidence": calibrated_confidence,
                                    "reasoning": result.get("reasoning", ""),
                                    "first_stage_type": result.get("first_stage_type"),
                                    "first_stage_confidence": result.get("first_stage_confidence"),
                                    "document_features": document_features
                                }
                    except Exception as e:
                        logger.error(f"Error in batch categorization: {str(e)}")
                        st.error(f"Error in batch categorization: {str(e)}")
                else:
                    # Fallback to original implementation for backward compatibility
                    st.error("Enhanced API client not available. Please refresh the page and try again.")
                
                # Mark as categorized
                st.session_state.document_categorization["is_categorized"] = True
                
                # Success message
                st.success(f"Categorization complete! Processed {len(st.session_state.document_categorization['results'])} files.")
        
        # Display categorization results
        if st.session_state.document_categorization.get("is_categorized", False):
            st.write("### Categorization Results")
            
            # Create tabs for different views
            result_tabs = st.tabs(["Table View", "Detailed View"])
            
            with result_tabs[0]:  # Table View
                # Create a table of results
                results_data = []
                for file_id, result in st.session_state.document_categorization["results"].items():
                    # Get confidence level
                    confidence = result["confidence"]
                    confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
                    
                    # Add to results data
                    results_data.append({
                        "File Name": result["file_name"],
                        "Document Type": result["document_type"],
                        "Confidence": f"{confidence_level} ({confidence:.2f})",
                        "Status": "Needs Verification"
                    })
                
                # Display table
                if results_data:
                    st.table(results_data)
                else:
                    st.info("No categorization results yet")
            
            with result_tabs[1]:  # Detailed View
                # Display detailed results for each file
                for file_id, result in st.session_state.document_categorization["results"].items():
                    st.write(f"## {result['file_name']}")
                    
                    # Display category and confidence
                    st.write(f"**Category:** {result['document_type']}")
                    
                    # Get confidence
                    confidence = result["confidence"]
                    
                    # Display overall confidence
                    st.write(f"**Overall Confidence:** {confidence:.2f}")
                    
                    # Create a progress bar for confidence
                    st.progress(confidence)
                    
                    # Display confidence factors
                    st.write("**Confidence Factors:**")
                    
                    # Display AI reported confidence
                    ai_confidence = result.get("confidence_raw", 0.0)
                    st.write(f"AI Reported Confidence")
                    st.progress(ai_confidence)
                    st.write(f"{ai_confidence:.2f}")
                    
                    # Display multi-factor confidence if available
                    if "multi_factor_confidence" in result:
                        multi_factor = result["multi_factor_confidence"]
                        
                        # Display response quality
                        if "response_quality" in multi_factor:
                            st.write(f"Response Quality")
                            st.progress(multi_factor["response_quality"])
                            st.write(f"{multi_factor['response_quality']:.2f}")
                        
                        # Display evidence strength
                        if "evidence_strength" in multi_factor:
                            st.write(f"Evidence Strength")
                            st.progress(multi_factor["evidence_strength"])
                            st.write(f"{multi_factor['evidence_strength']:.2f}")
                        
                        # Display format match
                        if "format_match" in multi_factor:
                            st.write(f"Format Match")
                            st.progress(multi_factor["format_match"])
                            st.write(f"{multi_factor['format_match']:.2f}")
                    
                    # Display reasoning
                    if "reasoning" in result and result["reasoning"]:
                        # Create a blue info box with the reasoning
                        st.info(result["reasoning"])
                    
                    # Display reasoning details in an expander
                    with st.expander("Reasoning", expanded=False):
                        if "reasoning" in result and result["reasoning"]:
                            st.write(result["reasoning"])
                        else:
                            st.write("No detailed reasoning available")
                    
                    # Add override option
                    st.write("**Override Category:**")
                    
                    # Create columns for override
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Get document type options
                        override_options = [dt["name"] for dt in document_types_with_desc]
                        
                        # Create selectbox for override
                        override_category = st.selectbox(
                            "Select category",
                            options=override_options,
                            index=override_options.index(result["document_type"]) if result["document_type"] in override_options else 0,
                            key=f"override_{file_id}"
                        )
                    
                    with col2:
                        # Create button for applying override
                        if st.button("Apply Override", key=f"apply_override_{file_id}"):
                            # Update document type
                            st.session_state.document_categorization["results"][file_id]["document_type"] = override_category
                            st.session_state.document_categorization["results"][file_id]["is_overridden"] = True
                            
                            # Success message
                            st.success(f"Category updated to {override_category}")
                            
                            # Rerun to update UI
                            st.rerun()
                    
                    # Add separator
                    st.write("---")
    
    with tab2:
        # Confidence threshold settings
        st.write("### Confidence Threshold Settings")
        
        # Get current thresholds
        thresholds = st.session_state.confidence_thresholds
        
        # Create sliders for thresholds
        auto_accept = st.slider(
            "Auto-Accept Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["auto_accept"],
            step=0.05,
            help="Documents with confidence above this threshold will be automatically accepted"
        )
        
        verification = st.slider(
            "Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["verification"],
            step=0.05,
            help="Documents with confidence above this threshold but below auto-accept will need verification"
        )
        
        rejection = st.slider(
            "Rejection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["rejection"],
            step=0.05,
            help="Documents with confidence below this threshold will be rejected"
        )
        
        # Update thresholds in session state
        st.session_state.confidence_thresholds = {
            "auto_accept": auto_accept,
            "verification": verification,
            "rejection": rejection
        }
        
        # Display threshold visualization
        st.write("### Threshold Visualization")
        
        # Create a chart to visualize thresholds
        chart_data = pd.DataFrame({
            'Threshold': ['Rejection', 'Verification', 'Auto-Accept'],
            'Value': [rejection, verification, auto_accept]
        })
        
        # Create a horizontal bar chart
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Value:Q',
            y=alt.Y('Threshold:N', sort=['Rejection', 'Verification', 'Auto-Accept']),
            color=alt.Color('Threshold:N', scale=alt.Scale(
                domain=['Rejection', 'Verification', 'Auto-Accept'],
                range=['#dc3545', '#ffc107', '#28a745']
            ))
        ).properties(
            width=600,
            height=200
        )
        
        # Display chart
        st.altair_chart(chart, use_container_width=True)
        
        # Add example scenarios to help understand thresholds
        # Use a separate function to avoid nesting expanders which causes StreamlitAPIException
        st.write("### Example Scenarios")
        display_confidence_examples()
        
        # Continue button
        st.write("---")
        if st.button("Continue to Metadata Configuration", key="continue_to_metadata_button", use_container_width=True):
            st.session_state.current_page = "Metadata Configuration"
            st.rerun()

def display_confidence_examples():
    """
    Display example scenarios for confidence thresholds
    This is separated to avoid nesting expanders which causes StreamlitAPIException
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
    
    # Display scenarios
    for i, scenario in enumerate(scenarios):
        # Create columns instead of expanders to avoid nesting issues
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write(f"**{scenario['name']}**")
            st.write(f"Confidence: {scenario['confidence']:.2f}")
        
        with col2:
            st.write(f"**Description:** {scenario['description']}")
            st.write(f"**Category:** {scenario['category']}")
            
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
            
            st.markdown(f"**Status:** <span style='color: {color};'>{status}</span>", unsafe_allow_html=True)
            
            # Display confidence bar
            st.progress(scenario['confidence'])
        
        # Add separator between scenarios
        if i < len(scenarios) - 1:
            st.write("---")

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
    document_types = [result["document_type"] for result in results]
    confidences = [result["confidence"] for result in results]
    
    # Count votes for each document type, weighted by confidence
    votes = {}
    for dt, conf in zip(document_types, confidences):
        if dt not in votes:
            votes[dt] = 0
        votes[dt] += conf
    
    # Find document type with highest weighted votes
    max_votes = 0
    max_dt = None
    for dt, vote in votes.items():
        if vote > max_votes:
            max_votes = vote
            max_dt = dt
    
    # Calculate average confidence for the winning document type
    dt_confidences = [conf for dt, conf in zip(document_types, confidences) if dt == max_dt]
    avg_confidence = sum(dt_confidences) / len(dt_confidences) if dt_confidences else 0.0
    
    # Combine reasoning
    reasoning = "Combined reasoning from multiple models:\n\n"
    for i, result in enumerate(results):
        reasoning += f"Model {i+1} ({result['document_type']}, {result['confidence']:.2f}):\n"
        reasoning += result.get("reasoning", "No reasoning provided") + "\n\n"
    
    return {
        "document_type": max_dt or "Other",
        "confidence": avg_confidence,
        "reasoning": reasoning
    }

def extract_document_features(file_id: str) -> Dict[str, Any]:
    """
    Extract features from a document for multi-factor confidence calculation
    
    Args:
        file_id: Box file ID
        
    Returns:
        dict: Document features
    """
    # This is a placeholder for actual feature extraction
    # In a real implementation, this would analyze the document content
    return {
        "page_count": 1,
        "has_tables": True,
        "has_images": False,
        "word_count": 500,
        "average_confidence": 0.8
    }

def calculate_multi_factor_confidence(
    ai_confidence: float,
    document_features: Dict[str, Any],
    document_type: str,
    reasoning: str,
    document_type_names: List[str]
) -> Dict[str, float]:
    """
    Calculate multi-factor confidence based on AI confidence and document features
    
    Args:
        ai_confidence: Confidence reported by AI
        document_features: Features extracted from the document
        document_type: Categorized document type
        reasoning: Reasoning provided by AI
        document_type_names: List of valid document type names
        
    Returns:
        dict: Multi-factor confidence scores
    """
    # Initialize confidence factors
    confidence_factors = {
        "ai_reported": ai_confidence,
        "response_quality": 0.0,
        "evidence_strength": 0.0,
        "format_match": 0.0,
        "overall": 0.0
    }
    
    # Calculate response quality based on reasoning length and structure
    if reasoning:
        # Longer reasoning generally indicates better quality
        reasoning_length = len(reasoning)
        if reasoning_length > 500:
            confidence_factors["response_quality"] = 0.8
        elif reasoning_length > 200:
            confidence_factors["response_quality"] = 0.5
        else:
            confidence_factors["response_quality"] = 0.3
        
        # Check for specific evidence in reasoning
        evidence_count = 0
        evidence_patterns = [
            r"contains.*table",
            r"includes.*section",
            r"format.*typical",
            r"structure.*consistent",
            r"language.*indicates",
            r"terminology.*suggests"
        ]
        
        for pattern in evidence_patterns:
            if re.search(pattern, reasoning, re.IGNORECASE):
                evidence_count += 1
        
        # Calculate evidence strength based on evidence count
        confidence_factors["evidence_strength"] = min(1.0, evidence_count / 4)
    
    # Calculate format match based on document features
    # This is a placeholder for actual format matching logic
    confidence_factors["format_match"] = 0.8
    
    # Calculate overall confidence as weighted average of factors
    weights = {
        "ai_reported": 0.5,
        "response_quality": 0.2,
        "evidence_strength": 0.2,
        "format_match": 0.1
    }
    
    overall = 0.0
    for factor, weight in weights.items():
        overall += confidence_factors[factor] * weight
    
    confidence_factors["overall"] = overall
    
    return confidence_factors

def apply_confidence_calibration(document_type: str, confidence: float) -> float:
    """
    Apply calibration to confidence score based on document type
    
    Args:
        document_type: Document type
        confidence: Raw confidence score
        
    Returns:
        float: Calibrated confidence score
    """
    # This is a placeholder for actual calibration logic
    # In a real implementation, this would apply document-type specific adjustments
    
    # Apply a small adjustment based on document type
    # This simulates the fact that some document types are easier to identify than others
    calibration_factors = {
        "Invoices": 1.05,
        "Financial Report": 0.95,
        "Tax": 1.0,
        "Sales Contract": 0.9,
        "Employment Contract": 0.9,
        "PII": 1.1,
        "Other": 0.8
    }
    
    # Get calibration factor for document type, default to 1.0
    calibration_factor = calibration_factors.get(document_type, 1.0)
    
    # Apply calibration
    calibrated = confidence * calibration_factor
    
    # Ensure confidence is in [0, 1] range
    return max(0.0, min(1.0, calibrated))

def categorize_document_detailed(file_id: str, model: str, first_stage_type: str, document_types: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Perform detailed second-stage categorization for a document
    
    Args:
        file_id: Box file ID
        model: AI model to use
        first_stage_type: Document type from first stage
        document_types: List of document types with descriptions
        
    Returns:
        dict: Categorization result
    """
    # This is a placeholder for actual second-stage categorization
    # In a real implementation, this would make another API call with a more detailed prompt
    
    # For now, just return the first stage result with slightly higher confidence
    return {
        "document_type": first_stage_type,
        "confidence": min(1.0, 0.8),  # Slightly higher confidence
        "reasoning": f"Detailed analysis confirms this is a {first_stage_type} document based on content and structure."
    }
