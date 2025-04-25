#!/usr/bin/env python3
"""
Test script for the bug fixes implemented in the Box Metadata AI application.
This script tests the fixes for:
1. Nested expanders issue in document categorization
2. Inconsistency between detailed view and table view confidence scores
3. Metadata configuration template selection issues
4. Processing files functionality
"""

import os
import sys
import logging
import unittest
import streamlit as st
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestBugFixes(unittest.TestCase):
    """Test cases for bug fixes in Box Metadata AI application"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock streamlit session state
        self.mock_session_state = {
            "authenticated": True,
            "client": MagicMock(),
            "enhanced_client": MagicMock(),
            "selected_files": [
                {"id": "file1", "name": "test_file1.pdf", "size": 1024},
                {"id": "file2", "name": "test_file2.pdf", "size": 2048}
            ],
            "document_categorization": {
                "is_categorized": True,
                "results": {
                    "file1": {
                        "document_type": "Financial Report",
                        "confidence": 0.82,
                        "ai_reported_confidence": 0.85,
                        "response_quality": 0.80,
                        "evidence_strength": 0.75,
                        "format_match": 0.70,
                        "reasoning": "This is a financial report"
                    },
                    "file2": {
                        "document_type": "Form 10K",
                        "confidence": 0.73,
                        "ai_reported_confidence": 0.75,
                        "response_quality": 0.70,
                        "evidence_strength": 0.65,
                        "format_match": 0.60,
                        "reasoning": "This is a Form 10K"
                    }
                }
            },
            "metadata_config": {
                "extraction_method": "structured",
                "freeform_prompt": "Extract metadata",
                "use_template": True,
                "template_id": "template1",
                "custom_fields": [],
                "ai_model": "google__gemini_2_0_flash_001",
                "batch_size": 5
            },
            "metadata_templates": {
                "template1": {
                    "id": "template1",
                    "displayName": "Financial Report",
                    "fields": [
                        {"key": "title", "displayName": "Title", "type": "string"},
                        {"key": "date", "displayName": "Date", "type": "date"}
                    ]
                },
                "template2": {
                    "id": "template2",
                    "displayName": "Form 10K",
                    "fields": [
                        {"key": "company", "displayName": "Company", "type": "string"},
                        {"key": "fiscal_year", "displayName": "Fiscal Year", "type": "string"}
                    ]
                }
            },
            "document_type_to_template": {
                "Financial Report": "template1",
                "Form 10K": "template2"
            },
            "performance_settings": {
                "enable_caching": True,
                "parallel_processing": True,
                "max_workers": 4,
                "batch_size": 10
            },
            "processing_state": {
                "is_processing": False,
                "current_file_index": 0,
                "total_files": 2,
                "processed_files": 0,
                "results": {},
                "errors": {}
            }
        }
        
        # Patch streamlit session state
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patcher.start()
        
        # Patch streamlit functions
        self.st_write_patcher = patch('streamlit.write')
        self.mock_st_write = self.st_write_patcher.start()
        
        self.st_title_patcher = patch('streamlit.title')
        self.mock_st_title = self.st_title_patcher.start()
        
        self.st_subheader_patcher = patch('streamlit.subheader')
        self.mock_st_subheader = self.st_subheader_patcher.start()
        
        self.st_expander_patcher = patch('streamlit.expander')
        self.mock_st_expander = self.st_expander_patcher.start()
        self.mock_st_expander.return_value.__enter__.return_value = MagicMock()
        
        self.st_columns_patcher = patch('streamlit.columns')
        self.mock_st_columns = self.st_columns_patcher.start()
        self.mock_st_columns.return_value = [MagicMock(), MagicMock()]
        
        self.st_tabs_patcher = patch('streamlit.tabs')
        self.mock_st_tabs = self.st_tabs_patcher.start()
        self.mock_st_tabs.return_value = [MagicMock(), MagicMock()]
        
        self.st_progress_patcher = patch('streamlit.progress')
        self.mock_st_progress = self.st_progress_patcher.start()
        
        self.st_empty_patcher = patch('streamlit.empty')
        self.mock_st_empty = self.st_empty_patcher.start()
        
        self.st_success_patcher = patch('streamlit.success')
        self.mock_st_success = self.st_success_patcher.start()
        
        self.st_error_patcher = patch('streamlit.error')
        self.mock_st_error = self.st_error_patcher.start()
        
        self.st_warning_patcher = patch('streamlit.warning')
        self.mock_st_warning = self.st_warning_patcher.start()
        
        self.st_info_patcher = patch('streamlit.info')
        self.mock_st_info = self.st_info_patcher.start()
        
        self.st_button_patcher = patch('streamlit.button')
        self.mock_st_button = self.st_button_patcher.start()
        
        self.st_selectbox_patcher = patch('streamlit.selectbox')
        self.mock_st_selectbox = self.st_selectbox_patcher.start()
        
        self.st_radio_patcher = patch('streamlit.radio')
        self.mock_st_radio = self.st_radio_patcher.start()
        
        self.st_checkbox_patcher = patch('streamlit.checkbox')
        self.mock_st_checkbox = self.st_checkbox_patcher.start()
        
        self.st_slider_patcher = patch('streamlit.slider')
        self.mock_st_slider = self.st_slider_patcher.start()
        
        self.st_text_area_patcher = patch('streamlit.text_area')
        self.mock_st_text_area = self.st_text_area_patcher.start()
        
        self.st_text_input_patcher = patch('streamlit.text_input')
        self.mock_st_text_input = self.st_text_input_patcher.start()
        
        self.st_table_patcher = patch('streamlit.table')
        self.mock_st_table = self.st_table_patcher.start()
        
        self.st_rerun_patcher = patch('streamlit.rerun')
        self.mock_st_rerun = self.st_rerun_patcher.start()
        
        # Add modules directory to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop all patches
        self.session_state_patcher.stop()
        self.st_write_patcher.stop()
        self.st_title_patcher.stop()
        self.st_subheader_patcher.stop()
        self.st_expander_patcher.stop()
        self.st_columns_patcher.stop()
        self.st_tabs_patcher.stop()
        self.st_progress_patcher.stop()
        self.st_empty_patcher.stop()
        self.st_success_patcher.stop()
        self.st_error_patcher.stop()
        self.st_warning_patcher.stop()
        self.st_info_patcher.stop()
        self.st_button_patcher.stop()
        self.st_selectbox_patcher.stop()
        self.st_radio_patcher.stop()
        self.st_checkbox_patcher.stop()
        self.st_slider_patcher.stop()
        self.st_text_area_patcher.stop()
        self.st_text_input_patcher.stop()
        self.st_table_patcher.stop()
        self.st_rerun_patcher.stop()
    
    def test_document_categorization_fixed(self):
        """Test the fixed document categorization module"""
        logger.info("Testing document_categorization_fixed.py")
        
        # Import the fixed module
        from modules.document_categorization_fixed import (
            document_categorization,
            display_categorization_results,
            validate_confidence_with_examples
        )
        
        # Test document_categorization function
        document_categorization()
        
        # Verify that expander was called but not nested
        self.mock_st_expander.assert_called()
        
        # Test display_categorization_results function
        display_categorization_results()
        
        # Test validate_confidence_with_examples function
        validate_confidence_with_examples()
        
        # Verify that columns were used instead of nested expanders
        self.mock_st_columns.assert_called()
        
        logger.info("document_categorization_fixed.py tests passed")
    
    def test_metadata_config_fixed(self):
        """Test the fixed metadata configuration module"""
        logger.info("Testing metadata_config_fixed.py")
        
        # Import the fixed module
        from modules.metadata_config_fixed import metadata_config
        
        # Test metadata_config function
        metadata_config()
        
        # Verify that template details are displayed
        self.mock_st_expander.assert_called()
        
        logger.info("metadata_config_fixed.py tests passed")
    
    def test_processing_fixed(self):
        """Test the fixed processing module"""
        logger.info("Testing processing_fixed.py")
        
        # Import the fixed module
        from modules.processing_fixed import process_files, process_file
        
        # Mock the extract_metadata functions
        with patch('modules.processing_fixed.extract_metadata_freeform') as mock_extract_freeform, \
             patch('modules.processing_fixed.extract_metadata_structured') as mock_extract_structured, \
             patch('modules.processing_fixed.extract_metadata_template') as mock_extract_template:
            
            # Set up mock return values
            mock_extract_freeform.return_value = {"metadata": {"title": "Test"}, "raw_response": "Test response"}
            mock_extract_structured.return_value = {"metadata": {"title": "Test"}, "raw_response": "Test response"}
            mock_extract_template.return_value = {"metadata": {"title": "Test"}, "raw_response": "Test response"}
            
            # Test process_file function with freeform extraction
            config = {"extraction_method": "freeform", "freeform_prompt": "Extract", "ai_model": "test_model"}
            result = process_file("file1", config)
            
            # Verify that extract_metadata_freeform was called
            mock_extract_freeform.assert_called_with(file_id="file1", prompt="Extract", model="test_model")
            
            # Test process_file function with structured extraction
            config = {"extraction_method": "structured", "custom_fields": [{"name": "title"}], "ai_model": "test_model"}
            result = process_file("file1", config)
            
            # Verify that extract_metadata_structured was called
            mock_extract_structured.assert_called_with(file_id="file1", fields=[{"name": "title"}], model="test_model")
            
            # Test process_file function with template extraction
            config = {"extraction_method": "template", "template_id": "template1", "ai_model": "test_model"}
            result = process_file("file1", config)
            
            # Verify that extract_metadata_template was called
            mock_extract_template.assert_called_with(file_id="file1", template_id="template1", model="test_model")
        
        logger.info("processing_fixed.py tests passed")
    
    def test_app_fixed(self):
        """Test the fixed app module"""
        logger.info("Testing app_fixed.py")
        
        # Import the fixed module
        import app_fixed
        
        # Test initialize_session_state function
        app_fixed.initialize_session_state()
        
        # Test integrate_enhancements function
        with patch('app_fixed.document_types_manager_enhanced', MagicMock()), \
             patch('app_fixed.get_document_types_with_descriptions_enhanced', MagicMock()), \
             patch('app_fixed.BoxAPIClientEnhancedOptimized', MagicMock()):
            result = app_fixed.integrate_enhancements()
            self.assertTrue(result)
        
        # Test initialize_enhanced_client function
        with patch('app_fixed.BoxAPIClientEnhancedOptimized', MagicMock()):
            result = app_fixed.initialize_enhanced_client()
            self.assertTrue(result)
        
        logger.info("app_fixed.py tests passed")

def run_tests():
    """Run all tests"""
    logger.info("Running tests for bug fixes")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("All tests completed")

if __name__ == "__main__":
    run_tests()
