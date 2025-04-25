"""
Updated document categorization module that uses the enhanced API client
and supports user-defined document types.
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
                                
                                # Store result
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": combined_result["document_type"],
                                    "confidence": combined_result["confidence"],
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
                                
                                # Store result
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": result["document_type"],
                                    "confidence": result["confidence"],
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
                    # Process each file individually
                    for file in st.session_state.selected_files:
                        file_id = file["id"]
                        file_name = file["name"]
                        
                        try:
                            if use_consensus and consensus_models:
                                # Multi-model consensus categorization
                                consensus_results = []
                                
                                # Create progress bar for models
                                model_progress = st.progress(0)
                                model_status = st.empty()
                                
                                # Process with each model
                                for i, model in enumerate(consensus_models):
                                    model_status.text(f"Processing with {model}...")
                                    result = categorize_document(file_id, model, document_types_with_desc)
                                    consensus_results.append(result)
                                    model_progress.progress((i + 1) / len(consensus_models))
                                
                                # Clear progress indicators
                                model_progress.empty()
                                model_status.empty()
                                
                                # Combine results using weighted voting
                                result = combine_categorization_results(consensus_results)
                                
                                # Add model details to reasoning
                                models_text = ", ".join(consensus_models)
                                result["reasoning"] = f"Consensus from models: {models_text}\n\n" + result["reasoning"]
                            else:
                                # First-stage categorization
                                result = categorize_document(file_id, selected_model, document_types_with_desc)
                                
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
                            
                            # Store result
                            st.session_state.document_categorization["results"][file_id] = {
                                "file_id": file_id,
                                "file_name": file_name,
                                "document_type": result["document_type"],
                                "confidence": result["confidence"],
                                "multi_factor_confidence": multi_factor_confidence,
                                "calibrated_confidence": calibrated_confidence,
                                "reasoning": result.get("reasoning", ""),
                                "first_stage_type": result.get("first_stage_type"),
                                "first_stage_confidence": result.get("first_stage_confidence"),
                                "document_features": document_features
                            }
                        except Exception as e:
                            logger.error(f"Error categorizing document {file_name}: {str(e)}")
                            st.session_state.document_categorization["errors"][file_id] = {
                                "file_id": file_id,
                                "file_name": file_name,
                                "error": str(e)
                            }
                
                # Apply confidence thresholds
                st.session_state.document_categorization["results"] = apply_confidence_thresholds(
                    st.session_state.document_categorization["results"]
                )
                
                # Mark as categorized
                st.session_state.document_categorization["is_categorized"] = True
                
                # Show success message
                num_processed = len(st.session_state.document_categorization["results"])
                num_errors = len(st.session_state.document_categorization["errors"])
                
                if num_errors == 0:
                    st.success(f"Categorization complete! Processed {num_processed} files.")
                else:
                    st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.")
        
        # Display categorization results
        if st.session_state.document_categorization["is_categorized"]:
            display_categorization_results(document_type_names)
    
    with tab2:
        # Confidence settings
        st.write("### Confidence Configuration")
        
        # Confidence threshold configuration
        configure_confidence_thresholds()
        
        # Confidence validation
        with st.expander("Confidence Validation", expanded=False):
            validate_confidence_with_examples()

def display_categorization_results(document_types=None):
    """
    Display categorization results with enhanced confidence visualization
    
    Args:
        document_types: List of document type names (optional)
    """
    st.write("### Categorization Results")
    
    # Get results from session state
    results = st.session_state.document_categorization["results"]
    
    if not results:
        st.info("No categorization results available.")
        return
    
    # If document_types not provided, try to get from session state
    if document_types is None:
        from modules.document_types_manager import get_document_type_names
        document_types = get_document_type_names()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Create a table of results with enhanced confidence display
        results_data = []
        for file_id, result in results.items():
            # Determine status based on thresholds
            status = result.get("status", "Review")
            
            # Determine confidence level and color
            confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
            if confidence >= 0.8:
                confidence_level = "High"
                confidence_color = "green"
            elif confidence >= 0.6:
                confidence_level = "Medium"
                confidence_color = "orange"
            else:
                confidence_level = "Low"
                confidence_color = "red"
            
            results_data.append({
                "File Name": result["file_name"],
                "Document Type": result["document_type"],
                "Confidence": f"<span style='color: {confidence_color};'>{confidence_level} ({confidence:.2f})</span>",
                "Status": status
            })
        
        if results_data:
            # Convert to DataFrame for display
            df = pd.DataFrame(results_data)
            
            # Display as HTML to preserve formatting
            st.markdown(
                df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
    
    with tab2:
        # Create detailed view with confidence visualization
        for file_id, result in results.items():
            with st.container():
                st.write(f"### {result['file_name']}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display document type and confidence
                    st.write(f"**Category:** {result['document_type']}")
                    
                    # Display confidence visualization
                    if "multi_factor_confidence" in result:
                        display_confidence_visualization(result["multi_factor_confidence"])
                    else:
                        # Fallback for results without multi-factor confidence
                        confidence = result.get("confidence", 0.0)
                        if confidence >= 0.8:
                            confidence_color = "#28a745"  # Green
                        elif confidence >= 0.6:
                            confidence_color = "#ffc107"  # Yellow
                        else:
                            confidence_color = "#dc3545"  # Red
                        
                        st.markdown(
                            f"""
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                    <div style="font-weight: bold; margin-right: 10px;">Confidence:</div>
                                    <div style="font-weight: bold; color: {confidence_color};">{confidence:.2f}</div>
                                </div>
                                <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden;">
                                    <div style="width: {confidence*100}%; background-color: {confidence_color}; height: 100%;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Display confidence explanation
                    if "multi_factor_confidence" in result:
                        explanations = get_confidence_explanation(
                            result["multi_factor_confidence"],
                            result["document_type"]
                        )
                        st.info(explanations["overall"])
                    
                    # Display reasoning
                    with st.expander("Reasoning", expanded=False):
                        st.write(result.get("reasoning", "No reasoning provided"))
                    
                    # Display first-stage results if available
                    if result.get("first_stage_type"):
                        with st.expander("First-Stage Results", expanded=False):
                            st.write(f"**First-stage category:** {result['first_stage_type']}")
                            st.write(f"**First-stage confidence:** {result['first_stage_confidence']:.2f}")
                
                with col2:
                    # Category override
                    st.write("**Override Category:**")
                    new_category = st.selectbox(
                        "Select category",
                        options=document_types,
                        index=document_types.index(result["document_type"]) if result["document_type"] in document_types else 0,
                        key=f"override_category_{file_id}"
                    )
                    
                    if st.button("Apply Override", key=f"apply_override_{file_id}"):
                        # Update the category
                        st.session_state.document_categorization["results"][file_id]["document_type"] = new_category
                        st.session_state.document_categorization["results"][file_id]["confidence"] = 1.0  # Max confidence for manual override
                        st.session_state.document_categorization["results"][file_id]["calibrated_confidence"] = 1.0
                        st.session_state.document_categorization["results"][file_id]["reasoning"] += f"\n\nManually overridden to: {new_category}"
                        
                        st.success(f"Category updated to {new_category}")
                        st.rerun()

def categorize_document(file_id: str, model: str, document_types: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Categorize a document using Box AI API
    
    Args:
        file_id: Box file ID
        model: AI model to use for categorization
        document_types: List of document types with name and description
        
    Returns:
        dict: Document categorization result
    """
    # Check if enhanced client is available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        # Use enhanced client
        return st.session_state.enhanced_client.categorize_document(file_id, document_types, model)
    
    # Fallback to original implementation for backward compatibility
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Create prompt with document types
    type_descriptions = "\n".join([f"- {dt['name']}: {dt['description']}" for dt in document_types])
    type_names = [dt['name'] for dt in document_types]
    
    prompt = (
        f"Analyze this document and categorize it into one of the following types:\n\n"
        f"{type_descriptions}\n\n"
        f"Provide your answer in the following format:\n"
        f"Category: [selected category]\n"
        f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
        f"Reasoning: [detailed explanation with specific evidence from the document]"
    )
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",  # Required parameter - single_item_qa or multiple_item_qa
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"  # Required parameter for basic_text
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making Box AI API call with request: {json.dumps(request_body)}")
        import requests
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Log response for debugging
        logger.info(f"Box AI API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        logger.info(f"Box AI API response data: {json.dumps(response_data)}")
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = parse_categorization_response(answer_text, type_names)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": "Other" if "Other" in type_names else type_names[-1],
            "confidence": 0.0,
            "reasoning": "Could not determine document type"
        }
    
    except Exception as e:
        logger.error(f"Error in Box AI API call: {str(e)}")
        raise Exception(f"Error categorizing document: {str(e)}")

def categorize_document_detailed(file_id: str, model: str, initial_category: str, document_types: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Perform a more detailed categorization for documents with low confidence
    
    Args:
        file_id: Box file ID
        model: AI model to use for categorization
        initial_category: Initial category from first-stage categorization
        document_types: List of document types with name and description
        
    Returns:
        dict: Document categorization result
    """
    # Check if enhanced client is available
    if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
        # Use enhanced client with a custom prompt for detailed analysis
        type_descriptions = "\n".join([f"- {dt['name']}: {dt['description']}" for dt in document_types])
        type_names = [dt['name'] for dt in document_types]
        
        # Create a more detailed prompt for second-stage analysis
        prompt = (
            f"Analyze this document in detail to determine its category. "
            f"The initial categorization suggested it might be '{initial_category}', but we need a more thorough analysis.\n\n"
            f"For each of the following categories, provide a score from 0-10 indicating how well the document matches that category, "
            f"along with specific evidence from the document:\n\n"
            f"{type_descriptions}\n\n"
            f"Then provide your final categorization in the following format:\n"
            f"Category: [selected category]\n"
            f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
            f"Reasoning: [detailed explanation with specific evidence from the document]"
        )
        
        # Use the ask method directly
        api_url = "ai/ask"
        
        # Construct request body
        request_body = {
            "mode": "single_item_qa",
            "prompt": prompt,
            "items": [
                {
                    "type": "file",
                    "id": file_id
                }
            ],
            "ai_agent": {
                "type": "ai_agent_ask",
                "basic_text": {
                    "model": model,
                    "mode": "default"
                }
            }
        }
        
        # Make API call
        response = st.session_state.enhanced_client.call_api(
            api_url,
            method="POST",
            data=request_body
        )
        
        # Process response
        if "answer" in response:
            answer_text = response["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = parse_categorization_response(answer_text, type_names)
            
            # Boost confidence slightly for detailed analysis
            confidence = min(confidence * 1.1, 1.0)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": initial_category,
            "confidence": 0.3,
            "reasoning": "Could not determine document type in detailed analysis"
        }
    
    # Fallback to original implementation for backward compatibility
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Get document type names
    type_names = [dt['name'] for dt in document_types]
    type_descriptions = "\n".join([f"- {dt['name']}: {dt['description']}" for dt in document_types])
    
    # Create a more detailed prompt for second-stage analysis
    prompt = (
        f"Analyze this document in detail to determine its category. "
        f"The initial categorization suggested it might be '{initial_category}', but we need a more thorough analysis.\n\n"
        f"For each of the following categories, provide a score from 0-10 indicating how well the document matches that category, "
        f"along with specific evidence from the document:\n\n"
        f"{type_descriptions}\n\n"
        f"Then provide your final categorization in the following format:\n"
        f"Category: [selected category]\n"
        f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
        f"Reasoning: [detailed explanation with specific evidence from the document]"
    )
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making detailed Box AI API call with request: {json.dumps(request_body)}")
        import requests
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = parse_categorization_response(answer_text, type_names)
            
            # Boost confidence slightly for detailed analysis
            # This reflects the more thorough analysis performed
            confidence = min(confidence * 1.1, 1.0)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": initial_category,
            "confidence": 0.3,
            "reasoning": "Could not determine document type in detailed analysis"
        }
    
    except Exception as e:
        logger.error(f"Error in detailed Box AI API call: {str(e)}")
        raise Exception(f"Error in detailed categorization: {str(e)}")

def parse_categorization_response(response_text: str, document_types: List[str]) -> Tuple[str, float, str]:
    """
    Parse the AI response to extract document type, confidence score, and reasoning
    
    Args:
        response_text: The AI response text
        document_types: List of valid document types
        
    Returns:
        tuple: (document_type, confidence, reasoning)
    """
    # Default values
    document_type = document_types[-1] if document_types else "Other"
    confidence = 0.5
    reasoning = response_text
    
    try:
        # Try to extract category using regex
        category_match = re.search(r"Category:\s*([^\n]+)", response_text, re.IGNORECASE)
        if category_match:
            category_text = category_match.group(1).strip()
            # Find the closest matching document type
            for dt in document_types:
                if dt.lower() in category_text.lower():
                    document_type = dt
                    break
        
        # Try to extract confidence using regex
        confidence_match = re.search(r"Confidence:\s*(0\.\d+|1\.0|1)", response_text, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            # If no explicit confidence, try to find confidence-related words
            confidence_words = {
                "very high": 0.9,
                "high": 0.8,
                "good": 0.7,
                "moderate": 0.6,
                "medium": 0.5,
                "low": 0.4,
                "very low": 0.3,
                "uncertain": 0.2
            }
            
            for word, value in confidence_words.items():
                if word in response_text.lower():
                    confidence = value
                    break
        
        # Try to extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*([^\n]+(?:\n[^\n]+)*)", response_text, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # If no document type was found in the structured response, try to find it in the full text
        if document_type == document_types[-1]:
            for dt in document_types:
                if dt.lower() in response_text.lower():
                    document_type = dt
                    break
        
        return document_type, confidence, reasoning
    
    except Exception as e:
        logger.error(f"Error parsing categorization response: {str(e)}")
        return document_type, confidence, reasoning

def extract_document_features(file_id: str) -> Dict[str, Any]:
    """
    Extract features from a document to aid in categorization
    
    Args:
        file_id: Box file ID
        
    Returns:
        dict: Document features
    """
    try:
        client = st.session_state.client
        file_info = client.file(file_id).get()
        
        features = {
            "extension": file_info.name.split(".")[-1].lower() if "." in file_info.name else "",
            "size_kb": file_info.size / 1024,
            "file_type": file_info.type
        }
        
        # Get text content preview if possible
        try:
            # Use Box API to get text representation
            text_content = ""
            
            # This is a simplified approach - in a real implementation,
            # you would use Box's content API to get text content
            # For now, we'll extract text from the file name and type
            text_content = f"{file_info.name} {file_info.type}"
            
            features["text_content"] = text_content
        except Exception as e:
            logger.warning(f"Could not extract text content: {str(e)}")
            features["text_content"] = ""
        
        return features
    except Exception as e:
        logger.error(f"Error extracting document features: {str(e)}")
        return {}

def calculate_multi_factor_confidence(
    ai_confidence: float,
    document_features: dict,
    category: str,
    response_text: str,
    document_types: List[str]
) -> dict:
    """
    Calculate a multi-factor confidence score based on various aspects
    
    Args:
        ai_confidence: The confidence score reported by the AI
        document_features: Features extracted from the document
        category: The assigned category
        response_text: The full AI response text
        document_types: List of valid document types
        
    Returns:
        dict: Multi-factor confidence scores
    """
    # Initialize confidence factors
    confidence_factors = {
        "ai_reported": ai_confidence,
        "response_quality": 0.0,
        "evidence_strength": 0.0,
        "format_match": 0.0,
        "overall": ai_confidence  # Default to AI reported confidence
    }
    
    # Calculate response quality factor
    # Based on length, structure, and completeness
    response_length = len(response_text)
    has_category = "category:" in response_text.lower()
    has_confidence = "confidence:" in response_text.lower()
    has_reasoning = "reasoning:" in response_text.lower()
    
    response_quality = 0.5  # Base value
    
    # Adjust based on length (longer responses tend to be more thorough)
    if response_length > 1000:
        response_quality += 0.3
    elif response_length > 500:
        response_quality += 0.2
    elif response_length > 200:
        response_quality += 0.1
    
    # Adjust based on structure
    if has_category and has_confidence and has_reasoning:
        response_quality += 0.2
    elif has_category and has_reasoning:
        response_quality += 0.1
    
    # Cap at 1.0
    confidence_factors["response_quality"] = min(response_quality, 1.0)
    
    # Calculate evidence strength factor
    # Based on specific mentions of evidence in the reasoning
    evidence_strength = 0.4  # Base value
    
    # Look for evidence markers
    evidence_markers = [
        "page", "section", "paragraph", "line", "table",
        "figure", "chart", "image", "header", "footer",
        "date", "amount", "number", "name", "address",
        "signature", "logo", "letterhead", "watermark"
    ]
    
    evidence_count = sum(1 for marker in evidence_markers if marker in response_text.lower())
    
    # Adjust based on evidence count
    if evidence_count > 5:
        evidence_strength += 0.5
    elif evidence_count > 3:
        evidence_strength += 0.3
    elif evidence_count > 1:
        evidence_strength += 0.1
    
    # Cap at 1.0
    confidence_factors["evidence_strength"] = min(evidence_strength, 1.0)
    
    # Calculate format match factor
    # Based on file extension and typical formats for the category
    format_match = 0.5  # Base value
    
    # Get file extension
    extension = document_features.get("extension", "").lower()
    
    # Define typical formats for each category
    # This is a simplified example - in a real implementation,
    # you would have more comprehensive mappings
    format_mappings = {
        "Sales Contract": ["pdf", "docx", "doc"],
        "Invoices": ["pdf", "xlsx", "xls", "csv"],
        "Tax": ["pdf", "xlsx", "xls", "csv"],
        "Financial Report": ["pdf", "xlsx", "xls", "pptx"],
        "Employment Contract": ["pdf", "docx", "doc"],
        "PII": ["pdf", "docx", "doc", "xlsx", "csv"],
        "Other": []
    }
    
    # Check if the category exists in our mappings
    if category in format_mappings:
        # Check if the extension matches typical formats for this category
        if extension in format_mappings[category]:
            format_match += 0.3
    
    # Check file size reasonability
    size_kb = document_features.get("size_kb", 0)
    
    # Very small files are less likely to be properly categorized
    if size_kb < 1:
        format_match -= 0.2
    elif size_kb < 10:
        format_match -= 0.1
    
    # Cap between 0.0 and 1.0
    confidence_factors["format_match"] = max(0.0, min(format_match, 1.0))
    
    # Calculate overall confidence
    # Weighted average of all factors
    weights = {
        "ai_reported": 0.5,
        "response_quality": 0.2,
        "evidence_strength": 0.2,
        "format_match": 0.1
    }
    
    overall_confidence = sum(
        confidence_factors[factor] * weight
        for factor, weight in weights.items()
    )
    
    # Cap between 0.0 and 1.0
    confidence_factors["overall"] = max(0.0, min(overall_confidence, 1.0))
    
    return confidence_factors

def apply_confidence_calibration(category: str, confidence: float) -> float:
    """
    Apply category-specific confidence calibration
    
    Args:
        category: Document category
        confidence: Raw confidence score
        
    Returns:
        float: Calibrated confidence score
    """
    # Define calibration factors for each category
    # These would ideally be derived from validation data
    calibration_factors = {
        "Sales Contract": 1.0,  # No adjustment
        "Invoices": 1.1,        # Boost confidence slightly
        "Tax": 0.9,             # Reduce confidence slightly
        "Financial Report": 1.0,
        "Employment Contract": 1.0,
        "PII": 0.8,             # Reduce confidence more
        "Other": 0.7            # Reduce confidence significantly
    }
    
    # Apply calibration if category exists in our factors
    if category in calibration_factors:
        calibrated = confidence * calibration_factors[category]
    else:
        # Default calibration for unknown categories
        calibrated = confidence * 0.9
    
    # Cap between 0.0 and 1.0
    return max(0.0, min(calibrated, 1.0))

def apply_confidence_thresholds(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Apply confidence thresholds to categorization results
    
    Args:
        results: Categorization results
        
    Returns:
        dict: Updated results with status
    """
    # Get thresholds from session state
    thresholds = st.session_state.confidence_thresholds
    
    # Apply thresholds to each result
    for file_id, result in results.items():
        # Use calibrated confidence if available, otherwise use raw confidence
        confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
        
        # Determine status based on thresholds
        if confidence >= thresholds["auto_accept"]:
            result["status"] = "Auto-Accepted"
        elif confidence >= thresholds["verification"]:
            result["status"] = "Needs Verification"
        elif confidence >= thresholds["rejection"]:
            result["status"] = "Low Confidence"
        else:
            result["status"] = "Rejected"
    
    return results

def configure_confidence_thresholds():
    """
    Configure confidence thresholds for categorization
    """
    # Get current thresholds
    thresholds = st.session_state.confidence_thresholds
    
    # Create sliders for each threshold
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_accept = st.slider(
            "Auto-Accept Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["auto_accept"],
            step=0.05,
            help="Results with confidence above this threshold will be automatically accepted"
        )
    
    with col2:
        verification = st.slider(
            "Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["verification"],
            step=0.05,
            help="Results with confidence above this threshold but below auto-accept will need verification"
        )
    
    with col3:
        rejection = st.slider(
            "Rejection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=thresholds["rejection"],
            step=0.05,
            help="Results with confidence below this threshold will be rejected"
        )
    
    # Validate thresholds
    if auto_accept < verification:
        st.error("Auto-Accept threshold must be greater than or equal to Verification threshold")
    
    if verification < rejection:
        st.error("Verification threshold must be greater than or equal to Rejection threshold")
    
    # Update thresholds if valid
    if auto_accept >= verification >= rejection:
        st.session_state.confidence_thresholds = {
            "auto_accept": auto_accept,
            "verification": verification,
            "rejection": rejection
        }
        
        # Display threshold visualization
        st.write("### Threshold Visualization")
        
        # Create a visualization of the thresholds
        threshold_data = [
            {"threshold": "Rejection", "value": rejection, "color": "#dc3545"},
            {"threshold": "Verification", "value": verification, "color": "#ffc107"},
            {"threshold": "Auto-Accept", "value": auto_accept, "color": "#28a745"}
        ]
        
        # Display as a horizontal bar
        st.markdown(
            f"""
            <div style="width: 100%; background-color: #f0f0f0; height: 30px; border-radius: 5px; overflow: hidden; position: relative; margin-bottom: 10px;">
                <div style="width: {rejection*100}%; background-color: #dc3545; height: 100%; position: absolute;"></div>
                <div style="width: {verification*100}%; background-color: #ffc107; height: 100%; position: absolute;"></div>
                <div style="width: {auto_accept*100}%; background-color: #28a745; height: 100%; position: absolute;"></div>
                <div style="width: 100%; height: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 10px; position: relative; color: white; font-weight: bold;">
                    <span>0.0</span>
                    <span>{rejection:.2f}</span>
                    <span>{verification:.2f}</span>
                    <span>{auto_accept:.2f}</span>
                    <span>1.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display legend
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #dc3545; margin-right: 5px; border-radius: 3px;"></div>
                    <span>Rejected</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ffc107; margin-right: 5px; border-radius: 3px;"></div>
                    <span>Low Confidence</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ffc107; margin-right: 5px; border-radius: 3px;"></div>
                    <span>Needs Verification</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #28a745; margin-right: 5px; border-radius: 3px;"></div>
                    <span>Auto-Accepted</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_confidence_visualization(confidence_factors: Dict[str, float]):
    """
    Display a visualization of confidence factors
    
    Args:
        confidence_factors: Dictionary of confidence factors
    """
    # Determine color for overall confidence
    overall = confidence_factors["overall"]
    if overall >= 0.8:
        overall_color = "#28a745"  # Green
    elif overall >= 0.6:
        overall_color = "#ffc107"  # Yellow
    else:
        overall_color = "#dc3545"  # Red
    
    # Display overall confidence
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="font-weight: bold; margin-right: 10px;">Overall Confidence:</div>
                <div style="font-weight: bold; color: {overall_color};">{overall:.2f}</div>
            </div>
            <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden;">
                <div style="width: {overall*100}%; background-color: {overall_color}; height: 100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display individual factors
    st.markdown("<div style='margin-top: 10px;'><strong>Confidence Factors:</strong></div>", unsafe_allow_html=True)
    
    # Define factors to display and their descriptions
    factors_to_display = {
        "ai_reported": "AI Reported Confidence",
        "response_quality": "Response Quality",
        "evidence_strength": "Evidence Strength",
        "format_match": "Format Match"
    }
    
    for factor, description in factors_to_display.items():
        if factor in confidence_factors:
            value = confidence_factors[factor]
            
            # Determine color
            if value >= 0.8:
                color = "#28a745"  # Green
            elif value >= 0.6:
                color = "#ffc107"  # Yellow
            else:
                color = "#dc3545"  # Red
            
            st.markdown(
                f"""
                <div style="margin-bottom: 5px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <div>{description}</div>
                        <div style="color: {color};">{value:.2f}</div>
                    </div>
                    <div style="width: 100%; background-color: #f0f0f0; height: 5px; border-radius: 3px; overflow: hidden;">
                        <div style="width: {value*100}%; background-color: {color}; height: 100%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def get_confidence_explanation(confidence_factors: Dict[str, float], category: str) -> Dict[str, str]:
    """
    Generate explanations for confidence factors
    
    Args:
        confidence_factors: Dictionary of confidence factors
        category: Document category
        
    Returns:
        dict: Explanations for each factor
    """
    explanations = {}
    
    # Overall confidence explanation
    overall = confidence_factors["overall"]
    if overall >= 0.8:
        explanations["overall"] = f"High confidence in '{category}' categorization. The AI model is very certain about this classification based on document content and structure."
    elif overall >= 0.6:
        explanations["overall"] = f"Moderate confidence in '{category}' categorization. The AI model found reasonable evidence for this classification, but some aspects of the document may be ambiguous."
    else:
        explanations["overall"] = f"Low confidence in '{category}' categorization. The AI model is uncertain about this classification, and the document may need manual review."
    
    # AI reported confidence explanation
    ai_reported = confidence_factors.get("ai_reported", 0.0)
    if ai_reported >= 0.8:
        explanations["ai_reported"] = "The AI model reported high confidence in its classification."
    elif ai_reported >= 0.6:
        explanations["ai_reported"] = "The AI model reported moderate confidence in its classification."
    else:
        explanations["ai_reported"] = "The AI model reported low confidence in its classification."
    
    # Response quality explanation
    response_quality = confidence_factors.get("response_quality", 0.0)
    if response_quality >= 0.8:
        explanations["response_quality"] = "The AI response was detailed and well-structured, indicating thorough analysis."
    elif response_quality >= 0.6:
        explanations["response_quality"] = "The AI response was adequately detailed, providing reasonable analysis."
    else:
        explanations["response_quality"] = "The AI response lacked detail or structure, suggesting limited analysis."
    
    # Evidence strength explanation
    evidence_strength = confidence_factors.get("evidence_strength", 0.0)
    if evidence_strength >= 0.8:
        explanations["evidence_strength"] = "The AI found strong evidence in the document supporting this classification."
    elif evidence_strength >= 0.6:
        explanations["evidence_strength"] = "The AI found moderate evidence in the document supporting this classification."
    else:
        explanations["evidence_strength"] = "The AI found limited evidence in the document supporting this classification."
    
    # Format match explanation
    format_match = confidence_factors.get("format_match", 0.0)
    if format_match >= 0.8:
        explanations["format_match"] = "The document format strongly matches typical formats for this category."
    elif format_match >= 0.6:
        explanations["format_match"] = "The document format reasonably matches typical formats for this category."
    else:
        explanations["format_match"] = "The document format does not match typical formats for this category."
    
    return explanations

def validate_confidence_with_examples():
    """
    Validate confidence thresholds with example scenarios
    """
    st.write("This section helps you understand how confidence thresholds affect categorization results.")
    
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
    for scenario in scenarios:
        with st.expander(f"{scenario['name']} ({scenario['confidence']:.2f} confidence)", expanded=False):
            st.write(f"**Description:** {scenario['description']}")
            st.write(f"**Category:** {scenario['category']}")
            st.write(f"**Confidence:** {scenario['confidence']:.2f}")
            
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
            st.markdown(
                f"""
                <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 10px;">
                    <div style="width: {scenario['confidence']*100}%; background-color: {color}; height: 100%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

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
    
    # Find the document type with the most votes
    if votes:
        winner = max(votes.items(), key=lambda x: x[1])
        document_type = winner[0]
        
        # Calculate combined confidence
        # Average of confidences for the winning document type
        winning_confidences = [conf for dt, conf in zip(document_types, confidences) if dt == document_type]
        confidence = sum(winning_confidences) / len(winning_confidences)
        
        # Boost confidence if multiple models agree
        if len(winning_confidences) > 1:
            confidence = min(confidence * 1.1, 1.0)
        
        # Combine reasoning from all results
        reasoning = "Combined results from multiple models:\n\n"
        for i, result in enumerate(results):
            reasoning += f"Model {i+1} ({result['document_type']}, {result['confidence']:.2f}):\n"
            reasoning += result.get("reasoning", "No reasoning provided") + "\n\n"
        
        return {
            "document_type": document_type,
            "confidence": confidence,
            "reasoning": reasoning
        }
    else:
        return {
            "document_type": "Other",
            "confidence": 0.0,
            "reasoning": "Could not determine document type"
        }
