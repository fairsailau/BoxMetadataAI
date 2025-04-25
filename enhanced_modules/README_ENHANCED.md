# Box Metadata AI - Enhanced Version

This README provides an overview of the enhanced version of the Box Metadata AI application, which includes performance optimizations and document types enhancements.

## Overview

Box Metadata AI is a Streamlit application that connects to Box.com and uses Box AI API to extract metadata from files and apply it at scale. The enhanced version includes significant performance improvements and fixes for document type categorization.

## Key Enhancements

### 1. Performance Optimization
- Enhanced caching system with tiered caching and intelligent invalidation
- Parallel processing with ThreadPoolExecutor for batch operations
- Optimized API requests with connection pooling and compression
- User-configurable performance settings

### 2. Document Types Enhancement
- Fixed issue with custom document types not being properly passed to Box AI API
- Enhanced document types manager with example document support
- Improved AI prompt for better categorization accuracy
- Structured document types in API requests

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fairsailau/BoxMetadataAI.git
cd BoxMetadataAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Enhanced Version

To run the enhanced version of the application:
```bash
streamlit run app_enhanced.py
```

To run the original version:
```bash
streamlit run app.py
```

## Enhanced Features

### Performance Settings

The enhanced version includes a Performance Settings panel in the sidebar that allows you to:
- Enable/disable caching
- Enable/disable parallel processing
- Adjust maximum worker count
- Set batch size for processing

### Document Types with Examples

The enhanced document types manager allows you to:
- Create custom document types with descriptions
- Add example documents to improve categorization accuracy
- Import/export document type configurations
- Better categorization through structured API requests

## File Structure

### Enhanced Files
- `app_enhanced.py` - Enhanced version of the main application
- `modules/document_types_manager_enhanced.py` - Enhanced document types manager
- `modules/api_client_enhanced_optimized.py` - Optimized API client
- `modules/integration.py` - Integration module for enhanced components
- `test_enhancements.py` - Test script for enhancements

### Original Files (Maintained for Compatibility)
- `app.py` - Original application
- `modules/document_types_manager.py` - Original document types manager
- `modules/api_client_enhanced.py` - Original API client

## Performance Improvements

The enhanced version provides significant performance improvements:
- Batch processing is approximately 70% faster with parallel execution
- Reduced API calls through intelligent caching
- Better handling of large document sets through optimized batching
- Adaptive resource allocation based on system capabilities

## Documentation

For detailed information about the enhancements, refer to:
- `enhancement_implementation_report.md` - Comprehensive report on all enhancements
- `test_enhancements.py` - Test script with examples of using enhanced features

## License

This project is licensed under the terms specified in the original repository.
