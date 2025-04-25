"""
Document types manager module for handling user-defined document types.
This module provides functionality for managing document types used in categorization.
"""

import streamlit as st
import logging
import json
import os
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def document_types_manager():
    """
    User interface for managing document types used in categorization
    """
    st.title("Document Types Configuration")
    
    # Initialize document types in session state if not exists
    if "document_types" not in st.session_state:
        st.session_state.document_types = [
            {
                "id": "sales_contract",
                "name": "Sales Contract",
                "description": "Legal agreements for sales of goods or services"
            },
            {
                "id": "invoices",
                "name": "Invoices",
                "description": "Bills for products or services rendered"
            },
            {
                "id": "tax",
                "name": "Tax",
                "description": "Tax-related documents including returns, forms, and notices"
            },
            {
                "id": "financial_report",
                "name": "Financial Report",
                "description": "Financial statements, balance sheets, income statements, and other financial documents"
            },
            {
                "id": "employment_contract",
                "name": "Employment Contract",
                "description": "Legal agreements between employers and employees"
            },
            {
                "id": "pii",
                "name": "PII",
                "description": "Documents containing personally identifiable information"
            },
            {
                "id": "other",
                "name": "Other",
                "description": "Documents that don't fit into any other category"
            }
        ]
    
    # Display introduction
    st.write("""
    Configure the document types that will be used for categorization. 
    These types will be used by the AI to categorize your documents.
    """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Add New Document Type", use_container_width=True):
            # Add a new empty document type
            new_id = f"type_{len(st.session_state.document_types) + 1}"
            st.session_state.document_types.append({
                "id": new_id,
                "name": "",
                "description": ""
            })
    
    with col2:
        if st.button("Import Types", use_container_width=True):
            st.session_state.show_import = True
    
    with col3:
        if st.button("Reset to Default", use_container_width=True):
            # Confirm reset
            st.session_state.confirm_reset = True
    
    # Import dialog
    if st.session_state.get("show_import", False):
        with st.expander("Import Document Types", expanded=True):
            st.write("Paste JSON configuration or upload a JSON file")
            
            import_method = st.radio(
                "Import method",
                options=["Paste JSON", "Upload File"],
                horizontal=True
            )
            
            if import_method == "Paste JSON":
                json_input = st.text_area("JSON Configuration", height=200)
                if st.button("Import from JSON"):
                    try:
                        imported_types = json.loads(json_input)
                        if validate_document_types(imported_types):
                            st.session_state.document_types = imported_types
                            st.success(f"Successfully imported {len(imported_types)} document types")
                            st.session_state.show_import = False
                            st.rerun()
                        else:
                            st.error("Invalid document types format. Each type must have id, name, and description.")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format. Please check your input.")
            else:
                uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
                if uploaded_file is not None:
                    try:
                        imported_types = json.load(uploaded_file)
                        if validate_document_types(imported_types):
                            st.session_state.document_types = imported_types
                            st.success(f"Successfully imported {len(imported_types)} document types")
                            st.session_state.show_import = False
                            st.rerun()
                        else:
                            st.error("Invalid document types format. Each type must have id, name, and description.")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format. Please check your file.")
            
            if st.button("Cancel Import"):
                st.session_state.show_import = False
                st.rerun()
    
    # Reset confirmation
    if st.session_state.get("confirm_reset", False):
        st.warning("Are you sure you want to reset to default document types? This will delete all custom types.")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, Reset", use_container_width=True):
                # Reset to default
                st.session_state.document_types = [
                    {
                        "id": "sales_contract",
                        "name": "Sales Contract",
                        "description": "Legal agreements for sales of goods or services"
                    },
                    {
                        "id": "invoices",
                        "name": "Invoices",
                        "description": "Bills for products or services rendered"
                    },
                    {
                        "id": "tax",
                        "name": "Tax",
                        "description": "Tax-related documents including returns, forms, and notices"
                    },
                    {
                        "id": "financial_report",
                        "name": "Financial Report",
                        "description": "Financial statements, balance sheets, income statements, and other financial documents"
                    },
                    {
                        "id": "employment_contract",
                        "name": "Employment Contract",
                        "description": "Legal agreements between employers and employees"
                    },
                    {
                        "id": "pii",
                        "name": "PII",
                        "description": "Documents containing personally identifiable information"
                    },
                    {
                        "id": "other",
                        "name": "Other",
                        "description": "Documents that don't fit into any other category"
                    }
                ]
                st.session_state.confirm_reset = False
                st.success("Document types reset to default")
                st.rerun()
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
    
    # Display document types
    st.subheader("Document Types")
    
    # Check if we have any document types
    if not st.session_state.document_types:
        st.warning("No document types defined. Add at least one document type to continue.")
    else:
        # Display each document type with edit/delete options
        for i, doc_type in enumerate(st.session_state.document_types):
            with st.container():
                col1, col2, col3 = st.columns([3, 5, 1])
                
                with col1:
                    # Name field
                    new_name = st.text_input(
                        "Name",
                        value=doc_type["name"],
                        key=f"name_{i}",
                        placeholder="Document Type Name"
                    )
                    st.session_state.document_types[i]["name"] = new_name
                
                with col2:
                    # Description field
                    new_description = st.text_input(
                        "Description",
                        value=doc_type["description"],
                        key=f"desc_{i}",
                        placeholder="Brief description of this document type"
                    )
                    st.session_state.document_types[i]["description"] = new_description
                
                with col3:
                    # Delete button (don't allow deleting the last type)
                    if len(st.session_state.document_types) > 1:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.document_types.pop(i)
                            st.rerun()
                
                st.divider()
    
    # Export option
    with st.expander("Export Document Types", expanded=False):
        st.write("Export your document types configuration as JSON")
        
        # Format the JSON with indentation for readability
        json_config = json.dumps(st.session_state.document_types, indent=2)
        st.code(json_config, language="json")
        
        # Download button
        if st.download_button(
            label="Download JSON Configuration",
            data=json_config,
            file_name="document_types.json",
            mime="application/json"
        ):
            st.success("Configuration downloaded")
    
    # Navigation buttons
    st.write("### Next Steps")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Back to File Browser", use_container_width=True):
            st.session_state.current_page = "File Browser"
            st.rerun()
    
    with col2:
        # Validate that we have at least one document type with name and description
        valid_types = [dt for dt in st.session_state.document_types if dt["name"] and dt["description"]]
        
        if valid_types:
            if st.button("Continue to Document Categorization", use_container_width=True):
                st.session_state.current_page = "Document Categorization"
                st.rerun()
        else:
            st.button("Continue to Document Categorization", use_container_width=True, disabled=True)
            st.warning("Please define at least one document type with name and description")

def validate_document_types(document_types):
    """
    Validate document types format
    
    Args:
        document_types: List of document types to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(document_types, list):
        return False
    
    for dt in document_types:
        if not isinstance(dt, dict):
            return False
        
        # Check required fields
        if "id" not in dt or "name" not in dt or "description" not in dt:
            return False
    
    return True

def get_document_type_names():
    """
    Get list of document type names from session state
    
    Returns:
        list: List of document type names
    """
    if "document_types" not in st.session_state:
        return ["Other"]
    
    return [dt["name"] for dt in st.session_state.document_types if dt["name"]]

def get_document_types_with_descriptions():
    """
    Get list of document types with descriptions from session state
    
    Returns:
        list: List of document types with name and description
    """
    if "document_types" not in st.session_state:
        return [{"name": "Other", "description": "Documents that don't fit into any other category"}]
    
    return [{"name": dt["name"], "description": dt["description"]} 
            for dt in st.session_state.document_types 
            if dt["name"] and dt["description"]]
