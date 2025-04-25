"""
Enhanced document types manager module for handling user-defined document types.
This module provides functionality for managing document types used in categorization,
with support for example documents to improve categorization accuracy.
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

def document_types_manager_enhanced():
    """
    Enhanced user interface for managing document types used in categorization
    with support for example documents
    """
    st.title("Document Types Configuration")
    
    # Initialize document types in session state if not exists
    if "document_types" not in st.session_state:
        st.session_state.document_types = [
            {
                "id": "sales_contract",
                "name": "Sales Contract",
                "description": "Legal agreements for sales of goods or services",
                "examples": []
            },
            {
                "id": "invoices",
                "name": "Invoices",
                "description": "Bills for products or services rendered",
                "examples": []
            },
            {
                "id": "tax",
                "name": "Tax",
                "description": "Tax-related documents including returns, forms, and notices",
                "examples": []
            },
            {
                "id": "financial_report",
                "name": "Financial Report",
                "description": "Financial statements, balance sheets, income statements, and other financial documents",
                "examples": []
            },
            {
                "id": "employment_contract",
                "name": "Employment Contract",
                "description": "Legal agreements between employers and employees",
                "examples": []
            },
            {
                "id": "pii",
                "name": "PII",
                "description": "Documents containing personally identifiable information",
                "examples": []
            },
            {
                "id": "other",
                "name": "Other",
                "description": "Documents that don't fit into any other category",
                "examples": []
            }
        ]
    
    # Display introduction
    st.write("""
    Configure the document types that will be used for categorization. 
    These types will be used by the AI to categorize your documents.
    
    For best results, add example documents for each type to improve categorization accuracy.
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
                "description": "",
                "examples": []
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
                        if validate_document_types_enhanced(imported_types):
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
                        if validate_document_types_enhanced(imported_types):
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
                        "description": "Legal agreements for sales of goods or services",
                        "examples": []
                    },
                    {
                        "id": "invoices",
                        "name": "Invoices",
                        "description": "Bills for products or services rendered",
                        "examples": []
                    },
                    {
                        "id": "tax",
                        "name": "Tax",
                        "description": "Tax-related documents including returns, forms, and notices",
                        "examples": []
                    },
                    {
                        "id": "financial_report",
                        "name": "Financial Report",
                        "description": "Financial statements, balance sheets, income statements, and other financial documents",
                        "examples": []
                    },
                    {
                        "id": "employment_contract",
                        "name": "Employment Contract",
                        "description": "Legal agreements between employers and employees",
                        "examples": []
                    },
                    {
                        "id": "pii",
                        "name": "PII",
                        "description": "Documents containing personally identifiable information",
                        "examples": []
                    },
                    {
                        "id": "other",
                        "name": "Other",
                        "description": "Documents that don't fit into any other category",
                        "examples": []
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
                col1, col2 = st.columns([3, 5])
                
                with col1:
                    # Name field
                    new_name = st.text_input(
                        "Name",
                        value=doc_type["name"],
                        key=f"name_{i}",
                        placeholder="Document Type Name"
                    )
                    st.session_state.document_types[i]["name"] = new_name
                    
                    # ID field (hidden, auto-generated)
                    if new_name and doc_type["id"] == f"type_{i+1}":
                        # Auto-generate ID from name if using default ID
                        new_id = new_name.lower().replace(" ", "_").replace("-", "_")
                        st.session_state.document_types[i]["id"] = new_id
                
                with col2:
                    # Description field
                    new_description = st.text_input(
                        "Description",
                        value=doc_type["description"],
                        key=f"desc_{i}",
                        placeholder="Brief description of this document type"
                    )
                    st.session_state.document_types[i]["description"] = new_description
                
                # Examples section
                with st.expander("Examples (Optional)", expanded=False):
                    st.write("Add example documents for this type to improve categorization accuracy")
                    
                    # Initialize examples if not exists
                    if "examples" not in doc_type:
                        doc_type["examples"] = []
                    
                    # Display existing examples
                    if doc_type["examples"]:
                        st.write(f"{len(doc_type['examples'])} example(s) added:")
                        for j, example in enumerate(doc_type["examples"]):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.text(example["name"])
                            with col2:
                                if st.button("Remove", key=f"remove_example_{i}_{j}"):
                                    doc_type["examples"].pop(j)
                                    st.rerun()
                    else:
                        st.info("No examples added yet. Examples help improve categorization accuracy.")
                    
                    # Add example button
                    if st.button("Add Example", key=f"add_example_{i}"):
                        st.session_state.show_file_picker = i
                
                # Delete button
                if len(st.session_state.document_types) > 1:
                    if st.button("Delete Type", key=f"delete_{i}"):
                        st.session_state.document_types.pop(i)
                        st.rerun()
                
                st.divider()
    
    # File picker for examples
    if "show_file_picker" in st.session_state and st.session_state.show_file_picker is not None:
        i = st.session_state.show_file_picker
        with st.expander("Select Example Document", expanded=True):
            # File browser component
            from modules.file_browser import mini_file_browser
            selected_file = mini_file_browser(key=f"example_file_browser_{i}")
            
            if selected_file:
                # Add to examples
                if "examples" not in st.session_state.document_types[i]:
                    st.session_state.document_types[i]["examples"] = []
                
                # Check if already exists
                if any(ex["id"] == selected_file["id"] for ex in st.session_state.document_types[i]["examples"]):
                    st.warning(f"{selected_file['name']} is already an example for this type")
                else:
                    st.session_state.document_types[i]["examples"].append({
                        "id": selected_file["id"],
                        "name": selected_file["name"]
                    })
                    
                    st.success(f"Added {selected_file['name']} as an example")
                
                st.session_state.show_file_picker = None
                st.rerun()
            
            if st.button("Cancel", key=f"cancel_file_picker_{i}"):
                st.session_state.show_file_picker = None
                st.rerun()
    
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

def validate_document_types_enhanced(document_types):
    """
    Validate document types format with enhanced validation
    
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
        
        # Validate examples if present
        if "examples" in dt:
            if not isinstance(dt["examples"], list):
                return False
            
            for example in dt["examples"]:
                if not isinstance(example, dict):
                    return False
                
                if "id" not in example or "name" not in example:
                    return False
    
    return True

def get_document_type_names_enhanced():
    """
    Get list of document type names from session state
    
    Returns:
        list: List of document type names
    """
    if "document_types" not in st.session_state:
        return ["Other"]
    
    return [dt["name"] for dt in st.session_state.document_types if dt["name"]]

def get_document_types_with_descriptions_enhanced():
    """
    Get list of document types with descriptions from session state
    
    Returns:
        list: List of document types with name, description, and examples
    """
    if "document_types" not in st.session_state:
        return [{"name": "Other", "description": "Documents that don't fit into any other category", "examples": []}]
    
    return [{
        "id": dt["id"],
        "name": dt["name"],
        "description": dt["description"],
        "examples": dt.get("examples", [])
    } for dt in st.session_state.document_types 
       if dt["name"] and dt["description"]]

# For backward compatibility
def document_types_manager():
    """
    Original document types manager function for backward compatibility
    """
    return document_types_manager_enhanced()

def get_document_type_names():
    """
    Original function for backward compatibility
    """
    return get_document_type_names_enhanced()

def get_document_types_with_descriptions():
    """
    Original function for backward compatibility
    """
    result = get_document_types_with_descriptions_enhanced()
    # Remove examples for backward compatibility
    for dt in result:
        if "examples" in dt:
            del dt["examples"]
    return result

def validate_document_types(document_types):
    """
    Original function for backward compatibility
    """
    return validate_document_types_enhanced(document_types)

# Mini file browser for selecting example documents
def mini_file_browser(key="mini_file_browser"):
    """
    Simplified file browser for selecting example documents
    
    Args:
        key: Unique key for Streamlit components
        
    Returns:
        dict: Selected file information or None
    """
    st.write("Select a file to use as an example:")
    
    # Get client
    if not hasattr(st.session_state, "client") or not st.session_state.client:
        st.error("Box client not available. Please authenticate first.")
        return None
    
    client = st.session_state.client
    
    # Initialize state
    if f"{key}_folder_id" not in st.session_state:
        st.session_state[f"{key}_folder_id"] = "0"  # Root folder
    
    current_folder_id = st.session_state[f"{key}_folder_id"]
    
    try:
        # Get current folder
        folder = client.folder(folder_id=current_folder_id).get()
        
        # Breadcrumb navigation
        breadcrumb = []
        if current_folder_id != "0":
            # Add root
            breadcrumb.append({"id": "0", "name": "All Files"})
            
            # Get path to current folder
            parent_folder = folder
            while parent_folder.id != "0":
                breadcrumb.append({"id": parent_folder.id, "name": parent_folder.name})
                parent_folder = client.folder(folder_id=parent_folder.parent.id).get()
            
            # Reverse to show root first
            breadcrumb.reverse()
        else:
            breadcrumb.append({"id": "0", "name": "All Files"})
        
        # Display breadcrumb
        breadcrumb_html = " / ".join([f"<a href='#' id='{item['id']}'>{item['name']}</a>" for item in breadcrumb])
        st.markdown(breadcrumb_html, unsafe_allow_html=True)
        
        # Handle breadcrumb clicks
        for item in breadcrumb:
            if st.button(item["name"], key=f"{key}_breadcrumb_{item['id']}"):
                st.session_state[f"{key}_folder_id"] = item["id"]
                st.rerun()
        
        # Get items in current folder
        items = folder.get_items(limit=100, offset=0)
        
        # Separate folders and files
        folders = []
        files = []
        
        for item in items:
            if item.type == "folder":
                folders.append({"id": item.id, "name": item.name, "type": "folder"})
            elif item.type == "file":
                files.append({"id": item.id, "name": item.name, "type": "file"})
        
        # Sort alphabetically
        folders.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())
        
        # Display folders
        if folders:
            st.write("### Folders")
            for folder in folders:
                if st.button(f"📁 {folder['name']}", key=f"{key}_folder_{folder['id']}"):
                    st.session_state[f"{key}_folder_id"] = folder["id"]
                    st.rerun()
        
        # Display files
        if files:
            st.write("### Files")
            selected_file = None
            
            for file in files:
                if st.button(f"📄 {file['name']}", key=f"{key}_file_{file['id']}"):
                    return file
        
        if not folders and not files:
            st.info("This folder is empty.")
            
        return None
    
    except Exception as e:
        st.error(f"Error browsing files: {str(e)}")
        return None
