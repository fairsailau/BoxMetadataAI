# Integration Guide for Enhanced Features

This guide explains how to integrate the enhanced features into the existing Box Metadata AI application.

## Overview

The enhanced version includes two major improvements:
1. Performance optimization
2. Document types enhancement with example support

This guide will help you integrate these enhancements into your existing codebase.

## Quick Integration

For the quickest integration, simply use the enhanced application file:

```bash
streamlit run app_enhanced.py
```

This file already includes all the necessary imports and integration code.

## Step-by-Step Integration

If you prefer to integrate the enhancements into your existing application, follow these steps:

### 1. Add Integration Module

First, add the `integration.py` module to your project:

```python
# modules/integration.py
import streamlit as st
import logging
import os
import re
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_enhancements():
    """
    Integrate the enhanced document types and optimized API client into the main application.
    This function should be called from app.py to enable the enhancements.
    """
    logger.info("Integrating enhanced document types and optimized API client...")
    
    # Import enhanced modules
    try:
        # Import enhanced document types manager
        from modules.document_types_manager_enhanced import (
            document_types_manager_enhanced,
            get_document_types_with_descriptions_enhanced
        )
        
        # Import optimized API client
        from modules.api_client_enhanced_optimized import (
            BoxAPIClientEnhancedOptimized
        )
        
        # Override original functions with enhanced versions
        import modules.document_types_manager
        modules.document_types_manager.document_types_manager = document_types_manager_enhanced
        modules.document_types_manager.get_document_types_with_descriptions = get_document_types_with_descriptions_enhanced
        
        # Add missing re import to api_client_enhanced_optimized
        import modules.api_client_enhanced_optimized
        if 're' not in modules.api_client_enhanced_optimized.__dict__:
            modules.api_client_enhanced_optimized.re = re
        
        logger.info("Successfully integrated enhanced modules")
        return True
    except Exception as e:
        logger.error(f"Error integrating enhancements: {str(e)}")
        return False

def initialize_enhanced_client():
    """
    Initialize the enhanced API client with optimizations.
    This function should be called after authentication to replace the standard client.
    """
    if not hasattr(st.session_state, "client") or not st.session_state.client:
        logger.warning("Cannot initialize enhanced client: No client available")
        return False
    
    try:
        # Import optimized API client
        from modules.api_client_enhanced_optimized import BoxAPIClientEnhancedOptimized
        
        # Create enhanced client
        st.session_state.enhanced_client = BoxAPIClientEnhancedOptimized(st.session_state.client)
        logger.info("Successfully initialized enhanced API client with optimizations")
        return True
    except Exception as e:
        logger.error(f"Error initializing enhanced client: {str(e)}")
        return False
```

### 2. Modify Your Main Application

Add the following imports to your main application file:

```python
from modules.integration import integrate_enhancements, initialize_enhanced_client
```

Then call the integration function at the beginning of your application:

```python
# Integrate enhanced modules
integrate_enhancements()
```

After authentication is successful, initialize the enhanced client:

```python
# Initialize enhanced client if authenticated and not already initialized
if st.session_state.authenticated and st.session_state.client and not st.session_state.enhanced_client:
    initialize_enhanced_client()
    logger.info("Initialized enhanced API client")
```

### 3. Add Performance Settings UI

Add the following code to your sidebar to enable performance settings:

```python
# Performance Settings
with st.expander("Performance Settings", expanded=False):
    st.session_state.performance_settings["enable_caching"] = st.checkbox(
        "Enable Caching", 
        value=st.session_state.performance_settings.get("enable_caching", True),
        key="enable_caching_checkbox",
        help="Cache API responses to improve performance"
    )
    
    st.session_state.performance_settings["parallel_processing"] = st.checkbox(
        "Enable Parallel Processing", 
        value=st.session_state.performance_settings.get("parallel_processing", True),
        key="parallel_processing_checkbox",
        help="Process multiple files in parallel"
    )
    
    st.session_state.performance_settings["max_workers"] = st.slider(
        "Max Parallel Workers",
        min_value=1,
        max_value=8,
        value=st.session_state.performance_settings.get("max_workers", 4),
        key="max_workers_slider",
        help="Maximum number of parallel workers (higher values may improve performance but increase resource usage)",
        disabled=not st.session_state.performance_settings["parallel_processing"]
    )
    
    st.session_state.performance_settings["batch_size"] = st.slider(
        "Batch Size",
        min_value=1,
        max_value=20,
        value=st.session_state.performance_settings.get("batch_size", 10),
        key="batch_size_slider",
        help="Number of files to process in each batch"
    )
```

### 4. Initialize Performance Settings

Add the following code to your session state initialization:

```python
# Performance settings
if not hasattr(st.session_state, "performance_settings"):
    st.session_state.performance_settings = {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "parallel_processing": True,
        "max_workers": 4,
        "batch_size": 10,
        "adaptive_retries": True
    }
```

## Using Enhanced Features

### Using Enhanced Document Types

The enhanced document types manager provides support for example documents:

```python
# Get document types with examples
document_types = get_document_types_with_descriptions_enhanced()

# Use enhanced API client for categorization
if hasattr(st.session_state, "enhanced_client") and st.session_state.enhanced_client:
    result = st.session_state.enhanced_client.categorize_document_enhanced(
        file_id, document_types, model="google__gemini_2_0_flash_001"
    )
else:
    # Fallback to original client
    result = st.session_state.client.categorize_document(
        file_id, document_types, model="google__gemini_2_0_flash_001"
    )
```

### Using Optimized Batch Processing

For batch operations, use the optimized batch processing:

```python
from modules.api_client_enhanced_optimized import optimized_batch_processing

# Define batch processor function
def process_batch(batch):
    batch_results = {}
    for file_id in batch:
        # Process each file
        batch_results[file_id] = process_file(file_id)
    return batch_results

# Use optimized batch processing
results = optimized_batch_processing(
    file_ids=file_ids,
    process_function=process_batch,
    batch_size=st.session_state.performance_settings.get("batch_size", 10),
    max_workers=st.session_state.performance_settings.get("max_workers", 4)
)
```

## Troubleshooting

### Missing 're' Module

If you encounter an error about the 're' module not being defined, add this import:

```python
import re
```

### Backward Compatibility

All enhanced modules maintain backward compatibility with the original functions. If you encounter any issues, you can always fall back to the original implementation:

```python
# Fallback to original document types manager
from modules.document_types_manager import document_types_manager, get_document_types_with_descriptions

# Fallback to original API client
from modules.api_client_enhanced import BoxAPIClientEnhanced
```

## Conclusion

By following this guide, you should be able to integrate all the enhanced features into your existing Box Metadata AI application. The enhancements provide significant performance improvements and better document type handling without breaking existing functionality.
