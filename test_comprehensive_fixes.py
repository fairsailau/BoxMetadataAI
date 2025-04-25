"""
Test script for comprehensive fixes in Box Metadata AI.
This script tests all the fixes implemented to address the reported issues.
"""

import streamlit as st
import logging
import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from modules.document_categorization_comprehensive_fix import (
    display_confidence_validation_examples,
    display_categorization_results,
    calculate_confidence_factors
)

from modules.metadata_config_comprehensive_fix import (
    configure_freeform_extraction_for_document_type,
    configure_structured_extraction_for_document_type
)

from modules.processing_comprehensive_fix import (
    process_single_file
)

class TestDocumentCategorization(unittest.TestCase):
    """Test cases for document categorization fixes"""
    
    @patch('streamlit.markdown')
    def test_display_confidence_validation_examples_no_nested_expanders(self, mock_markdown):
        """Test that display_confidence_validation_examples doesn't use nested expanders"""
        # Setup mock session state
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.8,
            "verification": 0.6,
            "rejection": 0.4
        }
        
        # Call the function
        display_confidence_validation_examples()
        
        # Verify markdown was called (for card-like display)
        mock_markdown.assert_called()
        
        # Check that the function doesn't use nested expanders
        # This is a bit tricky to test directly, but we can check that the implementation
        # uses the card-like approach with markdown instead of nested expanders
        
        # Get the first call to markdown
        first_call = mock_markdown.call_args_list[0][0][0]
        
        # Check that it contains the expected card-like HTML
        self.assertIn('<div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">', first_call)
    
    @patch('streamlit.table')
    @patch('streamlit.write')
    @patch('streamlit.progress')
    def test_display_categorization_results_consistent_confidence(self, mock_progress, mock_write, mock_table):
        """Test that display_categorization_results shows consistent confidence between views"""
        # Setup mock session state
        st.session_state.document_categorization = {
            "results": {
                "file1": {
                    "document_type": "Invoice",
                    "confidence": 0.84,
                    "ai_reported_confidence": 0.9,
                    "response_quality": 0.8,
                    "evidence_strength": 0.7,
                    "format_match": 0.8,
                    "reasoning": "This is an invoice"
                }
            }
        }
        
        st.session_state.selected_files = [
            {"id": "file1", "name": "test.pdf"}
        ]
        
        # Mock tabs
        with patch('streamlit.tabs') as mock_tabs:
            # Setup mock tab context managers
            mock_tab1 = MagicMock()
            mock_tab2 = MagicMock()
            mock_tabs.return_value = [mock_tab1, mock_tab2]
            
            # Call the function
            display_categorization_results()
            
            # Check that the same confidence value is used in both views
            # This is hard to test directly, but we can check that the confidence value
            # is formatted consistently in both views
            
            # Get the table data passed to the table function
            table_data = mock_table.call_args_list[0][0][0]
            
            # Check that the confidence in the table includes the raw value
            self.assertIn("(0.84)", table_data[0]["Confidence"])
            
            # Check that the overall confidence in the detailed view is the raw value
            # This is called in the context of the second tab
            with mock_tab2:
                mock_write.assert_any_call("**Overall Confidence:** 0.84")
    
    def test_calculate_confidence_factors(self):
        """Test that calculate_confidence_factors works correctly"""
        # Test input
        result = {
            "confidence": 0.85,
            "reasoning": "This document contains invoice elements like line items, totals, and company information.",
            "raw_response": '{"document_type": "Invoice", "confidence": 0.85, "reasoning": "..."}'
        }
        
        # Call the function
        factors = calculate_confidence_factors(result, "Invoice", "A document for billing")
        
        # Check the results
        self.assertEqual(factors["ai_reported_confidence"], 0.85)
        self.assertGreaterEqual(factors["response_quality"], 0.8)
        self.assertGreaterEqual(factors["evidence_strength"], 0.3)
        self.assertGreaterEqual(factors["format_match"], 0.3)

class TestMetadataConfiguration(unittest.TestCase):
    """Test cases for metadata configuration fixes"""
    
    @patch('streamlit.text_area')
    def test_configure_freeform_extraction_for_document_type(self, mock_text_area):
        """Test that configure_freeform_extraction_for_document_type works correctly"""
        # Setup mock session state
        st.session_state.metadata_config = {
            "document_type_prompts": {
                "Invoice": "Extract invoice metadata"
            },
            "freeform_prompt": "Extract metadata"
        }
        
        # Call the function
        configure_freeform_extraction_for_document_type("Invoice")
        
        # Check that the text area was called with the correct prompt
        mock_text_area.assert_called_with(
            "Prompt for Invoice",
            value="Extract invoice metadata",
            height=150,
            key="prompt_Invoice",
            help="Customize the prompt for Invoice documents"
        )
    
    @patch('streamlit.selectbox')
    def test_configure_structured_extraction_for_document_type(self, mock_selectbox):
        """Test that configure_structured_extraction_for_document_type works correctly"""
        # Setup mock session state
        st.session_state.document_type_to_template = {
            "Invoice": "template1"
        }
        
        st.session_state.metadata_templates = {
            "template1": {
                "displayName": "Invoice Template",
                "id": "template1",
                "fields": [
                    {"displayName": "Invoice Number", "type": "string"}
                ]
            }
        }
        
        # Call the function with patch for write
        with patch('streamlit.write'):
            configure_structured_extraction_for_document_type("Invoice")
        
        # Check that the selectbox was called with the correct options
        mock_selectbox.assert_called()
        
        # The first call should be for the template selection
        call_args = mock_selectbox.call_args_list[0]
        self.assertEqual(call_args[1]["key"], "template_Invoice")

class TestProcessing(unittest.TestCase):
    """Test cases for processing fixes"""
    
    def test_process_single_file_with_document_type(self):
        """Test that process_single_file handles document-type specific extraction methods"""
        # Setup mock session state
        st.session_state.document_categorization = {
            "is_categorized": True,
            "results": {
                "file1": {
                    "document_type": "Invoice"
                }
            }
        }
        
        st.session_state.document_type_extraction_methods = {
            "Invoice": "freeform"
        }
        
        st.session_state.metadata_config = {
            "document_type_prompts": {
                "Invoice": "Extract invoice metadata"
            },
            "ai_model": "google__gemini_2_0_flash_001",
            "apply_metadata": False
        }
        
        # Mock the client
        st.session_state.client = MagicMock()
        
        # Mock the extract_metadata_freeform function
        with patch('modules.processing_comprehensive_fix.extract_metadata_freeform') as mock_extract:
            # Setup mock return value
            mock_extract.return_value = {
                "metadata": {
                    "invoice_number": "INV-123",
                    "date": "2025-04-25"
                }
            }
            
            # Call the function
            file = {"id": "file1", "name": "invoice.pdf"}
            file_id, result, error = process_single_file(file)
            
            # Check the results
            self.assertEqual(file_id, "file1")
            self.assertIsNotNone(result)
            self.assertIsNone(error)
            
            # Check that extract_metadata_freeform was called with the correct parameters
            mock_extract.assert_called_with(
                file_id="file1",
                prompt="Extract invoice metadata",
                model="google__gemini_2_0_flash_001"
            )

def run_tests():
    """Run all tests"""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDocumentCategorization))
    suite.addTest(unittest.makeSuite(TestMetadataConfiguration))
    suite.addTest(unittest.makeSuite(TestProcessing))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
