"""
Metadata configuration module for Box Metadata AI.
This module provides functionality for configuring metadata extraction parameters.
COMPREHENSIVE FIX: Added per-document-type extraction method selection and improved template mapping.
"""

import streamlit as st
import logging
import json
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def metadata_config():
    """
    Configure metadata extraction parameters
    """
    st.title("Metadata Configuration")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_config"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Check if document categorization has been performed
    has_categorization = (
        hasattr(st.session_state, "document_categorization") and 
        st.session_state.document_categorization.get("is_categorized", False)
    )
    
    # Display document categorization results if available
    if has_categorization:
        st.subheader("Document Categorization Results")
        
        # Create a table of document types
        categorization_data = []
        for file in st.session_state.selected_files:
            file_id = file["id"]
            file_name = file["name"]
            
            # Get document type from categorization results
            document_type = "Not categorized"
            if file_id in st.session_state.document_categorization["results"]:
                document_type = st.session_state.document_categorization["results"][file_id]["document_type"]
            
            categorization_data.append({
                "File Name": file_name,
                "Document Type": document_type
            })
        
        # Display table
        st.table(categorization_data)
    else:
        st.info("Document categorization has not been performed. You can categorize documents in the Document Categorization page.")
        if st.button("Go to Document Categorization", key="go_to_doc_cat_button"):
            st.session_state.current_page = "Document Categorization"
            st.rerun()
    
    # Initialize document type extraction methods if not exists
    if not hasattr(st.session_state, "document_type_extraction_methods"):
        st.session_state.document_type_extraction_methods = {}
    
    # Global extraction method selection (only shown if no categorization)
    if not has_categorization:
        st.subheader("Extraction Method")
        
        # Ensure extraction_method is initialized in metadata_config
        if "extraction_method" not in st.session_state.metadata_config:
            st.session_state.metadata_config["extraction_method"] = "freeform"
            
        extraction_method = st.radio(
            "Select extraction method",
            ["Freeform", "Structured"],
            index=0 if st.session_state.metadata_config["extraction_method"] == "freeform" else 1,
            key="extraction_method_radio",
            help="Choose between freeform extraction (free text) or structured extraction (with template)"
        )
        
        # Update extraction method in session state
        st.session_state.metadata_config["extraction_method"] = extraction_method.lower()
        
        # Configure based on selected method
        if extraction_method == "Freeform":
            configure_freeform_extraction()
        else:
            configure_structured_extraction()
    else:
        # Get unique document types
        document_types = set()
        for file_id, result in st.session_state.document_categorization["results"].items():
            document_types.add(result["document_type"])
        
        # Create a list of document types to ensure consistent order
        document_types_list = sorted(list(document_types))
        
        # Per-document type configuration
        st.subheader("Document Type Configuration")
        st.info("Configure extraction method and settings for each document type.")
        
        # Configure each document type
        for doc_type in document_types_list:
            with st.expander(f"Configuration for {doc_type}", expanded=True):
                # Get current extraction method for document type
                current_method = st.session_state.document_type_extraction_methods.get(doc_type, "freeform")
                
                # Extraction method selection for this document type
                extraction_method = st.radio(
                    f"Select extraction method for {doc_type}",
                    ["Freeform", "Structured"],
                    index=0 if current_method == "freeform" else 1,
                    key=f"extraction_method_{doc_type.replace(' ', '_').lower()}",
                    help=f"Choose extraction method for {doc_type} documents"
                )
                
                # Update extraction method in session state
                st.session_state.document_type_extraction_methods[doc_type] = extraction_method.lower()
                
                # Configure based on selected method
                if extraction_method == "Freeform":
                    configure_freeform_extraction_for_document_type(doc_type)
                else:
                    configure_structured_extraction_for_document_type(doc_type)
    
    # AI model selection
    st.subheader("AI Model Selection")
    
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
        "Select AI Model",
        options=ai_models,
        index=ai_models.index(st.session_state.metadata_config["ai_model"]) if st.session_state.metadata_config["ai_model"] in ai_models else 0,
        key="ai_model_selectbox",
        help="Choose the AI model to use for metadata extraction"
    )
    
    # Update AI model in session state
    st.session_state.metadata_config["ai_model"] = selected_model
    
    # Batch processing configuration
    st.subheader("Batch Processing Configuration")
    
    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=10,
        value=st.session_state.metadata_config["batch_size"],
        step=1,
        key="batch_size_slider",
        help="Number of files to process in parallel"
    )
    
    # Update batch size in session state
    st.session_state.metadata_config["batch_size"] = batch_size
    
    # Continue button
    st.write("---")
    if st.button("Continue to Process Files", key="continue_to_process_button", use_container_width=True):
        st.session_state.current_page = "Process Files"
        st.rerun()

def configure_freeform_extraction():
    """
    Configure freeform extraction settings
    """
    st.subheader("Freeform Extraction Configuration")
    
    # Freeform prompt
    freeform_prompt = st.text_area(
        "Freeform prompt",
        value=st.session_state.metadata_config.get("freeform_prompt", "Extract key metadata from this document including title, date, author, and any other relevant information. Format the response as key-value pairs."),
        height=150,
        key="freeform_prompt_textarea",
        help="Prompt for freeform extraction. Be specific about what metadata to extract."
    )
    
    # Update freeform prompt in session state
    st.session_state.metadata_config["freeform_prompt"] = freeform_prompt

def configure_freeform_extraction_for_document_type(doc_type):
    """
    Configure freeform extraction settings for a specific document type
    
    Args:
        doc_type: Document type
    """
    # Initialize document type prompts if not exists
    if "document_type_prompts" not in st.session_state.metadata_config:
        st.session_state.metadata_config["document_type_prompts"] = {}
    
    # Get current prompt for document type
    current_prompt = st.session_state.metadata_config["document_type_prompts"].get(
        doc_type, st.session_state.metadata_config.get("freeform_prompt", "Extract key metadata from this document including title, date, author, and any other relevant information. Format the response as key-value pairs.")
    )
    
    # Display prompt input
    doc_type_prompt = st.text_area(
        f"Prompt for {doc_type}",
        value=current_prompt,
        height=150,
        key=f"prompt_{doc_type.replace(' ', '_').lower()}",
        help=f"Customize the prompt for {doc_type} documents"
    )
    
    # Update prompt in session state
    st.session_state.metadata_config["document_type_prompts"][doc_type] = doc_type_prompt

def configure_structured_extraction():
    """
    Configure structured extraction settings
    """
    st.subheader("Structured Extraction Configuration")
    
    # Check if metadata templates are available
    if not hasattr(st.session_state, "metadata_templates") or not st.session_state.metadata_templates:
        st.warning("No metadata templates available. Please refresh templates in the sidebar.")
        return
    
    # Get available templates
    templates = st.session_state.metadata_templates
    
    # Create template options
    template_options = [("", "None - Use custom fields")]
    for template_id, template in templates.items():
        template_options.append((template_id, template["displayName"]))
    
    # Template selection
    selected_template_name = st.selectbox(
        "Select a metadata template",
        options=[option[1] for option in template_options],
        index=0,
        key="template_selectbox",
        help="Select a metadata template to use for structured extraction"
    )
    
    # Find template ID from selected name
    selected_template_id = ""
    for template_id, template_name in template_options:
        if template_name == selected_template_name:
            selected_template_id = template_id
            break
    
    # Update template ID in session state
    st.session_state.metadata_config["template_id"] = selected_template_id
    st.session_state.metadata_config["use_template"] = (selected_template_id != "")
    
    # Display template details if selected
    if selected_template_id and selected_template_id in templates:
        template = templates[selected_template_id]
        
        st.write("#### Template Details")
        st.write(f"**Name:** {template['displayName']}")
        st.write(f"**ID:** {template['id']}")
        
        # Display fields
        st.write("**Fields:**")
        for field in template["fields"]:
            st.write(f"- {field['displayName']} ({field['type']})")
    
    # Custom fields if no template selected
    if not selected_template_id:
        configure_custom_fields()

def configure_structured_extraction_for_document_type(doc_type):
    """
    Configure structured extraction settings for a specific document type
    
    Args:
        doc_type: Document type
    """
    # Initialize document type to template mapping if not exists
    if not hasattr(st.session_state, "document_type_to_template"):
        from modules.metadata_template_retrieval import initialize_template_state
        initialize_template_state()
    
    # Check if metadata templates are available
    if not hasattr(st.session_state, "metadata_templates") or not st.session_state.metadata_templates:
        st.warning("No metadata templates available. Please refresh templates in the sidebar.")
        return
    
    # Get available templates
    templates = st.session_state.metadata_templates
    
    # Create template options
    template_options = [("", "None - Use custom fields")]
    for template_id, template in templates.items():
        template_options.append((template_id, template["displayName"]))
    
    # Get current template for document type
    current_template_id = st.session_state.document_type_to_template.get(doc_type, "")
    
    # Find index of current template in options
    selected_index = 0
    for i, (template_id, _) in enumerate(template_options):
        if template_id == current_template_id:
            selected_index = i
            break
    
    # Display template selection
    selected_template = st.selectbox(
        f"Template for {doc_type}",
        options=[option[1] for option in template_options],
        index=selected_index,
        key=f"template_{doc_type.replace(' ', '_').lower()}",
        help=f"Select a metadata template for {doc_type} documents"
    )
    
    # Find template ID from selected name
    selected_template_id = ""
    for template_id, template_name in template_options:
        if template_name == selected_template:
            selected_template_id = template_id
            break
    
    # Update template in session state
    st.session_state.document_type_to_template[doc_type] = selected_template_id
    
    # Display template details if selected
    if selected_template_id and selected_template_id in templates:
        template = templates[selected_template_id]
        
        st.write("#### Template Details")
        st.write(f"**Name:** {template['displayName']}")
        st.write(f"**ID:** {template['id']}")
        
        # Display fields
        st.write("**Fields:**")
        for field in template["fields"]:
            st.write(f"- {field['displayName']} ({field['type']})")
    
    # Custom fields if no template selected
    if not selected_template_id:
        configure_custom_fields_for_document_type(doc_type)

def configure_custom_fields():
    """
    Configure custom fields for structured extraction
    """
    st.write("#### Custom Fields")
    st.write("Define custom fields for structured extraction")
    
    # Initialize custom fields if not exists
    if "custom_fields" not in st.session_state.metadata_config:
        st.session_state.metadata_config["custom_fields"] = []
    
    # Display existing custom fields
    for i, field in enumerate(st.session_state.metadata_config["custom_fields"]):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            field_name = st.text_input(
                "Field Name",
                value=field["name"],
                key=f"field_name_{i}",
                help="Name of the custom field"
            )
        
        with col2:
            field_type = st.selectbox(
                "Field Type",
                options=["string", "number", "date", "enum"],
                index=["string", "number", "date", "enum"].index(field["type"]),
                key=f"field_type_{i}",
                help="Type of the custom field"
            )
        
        with col3:
            if st.button("Remove", key=f"remove_field_{i}"):
                st.session_state.metadata_config["custom_fields"].pop(i)
                st.rerun()
        
        # Update field in session state
        st.session_state.metadata_config["custom_fields"][i]["name"] = field_name
        st.session_state.metadata_config["custom_fields"][i]["type"] = field_type
    
    # Add new field button
    if st.button("Add Field", key="add_field_button"):
        st.session_state.metadata_config["custom_fields"].append({
            "name": f"Field {len(st.session_state.metadata_config['custom_fields']) + 1}",
            "type": "string"
        })
        st.rerun()

def configure_custom_fields_for_document_type(doc_type):
    """
    Configure custom fields for structured extraction for a specific document type
    
    Args:
        doc_type: Document type
    """
    st.write(f"#### Custom Fields for {doc_type}")
    st.write(f"Define custom fields for {doc_type} documents")
    
    # Initialize document type custom fields if not exists
    if "document_type_custom_fields" not in st.session_state.metadata_config:
        st.session_state.metadata_config["document_type_custom_fields"] = {}
    
    # Initialize custom fields for document type if not exists
    if doc_type not in st.session_state.metadata_config["document_type_custom_fields"]:
        st.session_state.metadata_config["document_type_custom_fields"][doc_type] = []
    
    # Get custom fields for document type
    custom_fields = st.session_state.metadata_config["document_type_custom_fields"][doc_type]
    
    # Display existing custom fields
    for i, field in enumerate(custom_fields):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            field_name = st.text_input(
                "Field Name",
                value=field["name"],
                key=f"field_name_{doc_type.replace(' ', '_').lower()}_{i}",
                help="Name of the custom field"
            )
        
        with col2:
            field_type = st.selectbox(
                "Field Type",
                options=["string", "number", "date", "enum"],
                index=["string", "number", "date", "enum"].index(field["type"]),
                key=f"field_type_{doc_type.replace(' ', '_').lower()}_{i}",
                help="Type of the custom field"
            )
        
        with col3:
            if st.button("Remove", key=f"remove_field_{doc_type.replace(' ', '_').lower()}_{i}"):
                custom_fields.pop(i)
                st.rerun()
        
        # Update field in session state
        custom_fields[i]["name"] = field_name
        custom_fields[i]["type"] = field_type
    
    # Add new field button
    if st.button("Add Field", key=f"add_field_button_{doc_type.replace(' ', '_').lower()}"):
        custom_fields.append({
            "name": f"Field {len(custom_fields) + 1}",
            "type": "string"
        })
        st.rerun()
    
    # Update custom fields in session state
    st.session_state.metadata_config["document_type_custom_fields"][doc_type] = custom_fields
