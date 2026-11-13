"""
Unit testing suite for the Identity Verification Utility.
Validates the AAMVA barcode parser and routing logic without triggering the hardware camera.
"""

import unittest
from unittest.mock import patch
from user_identaty_util import get_identaty, _parse_aamva_barcode

class TestUserIdentityUtil(unittest.TestCase):

    def test_parse_aamva_barcode_valid(self):
        """
        Test that the regex parser correctly extracts data from a valid 
        Ontario driver's license PDF417 payload (containing tags like <LF>).
        """
        # Mock payload similar to the zxing-cpp output from an Ontario license
        mock_payload = (
            "@<LF><RS><CR>ANSI 636012090002DL00410227ZO02680058DLDCAG<LF>"
            "DCSGALSTYAN<LF>DACSERZH<LF>DAD<LF>DBD20240529<LF>DBB20060413<LF>"
            "DBC1<LF>DAYUNK<LF>DAU175 cm<LF>"
        )
        
        expected_data = {
            "first_name": "SERZH",
            "last_name": "GALSTYAN",
            "birth_date": "2006-04-13"
        }
        
        result = _parse_aamva_barcode(mock_payload)
        self.assertEqual(result, expected_data)

    def test_parse_aamva_barcode_missing_fields(self):
        """
        Test the parser's fallback behavior when the payload is missing expected tags.
        """
        mock_payload = "ANSI 123456<LF>DCSDOE<LF>" # Missing DAC (First Name) and DBB (DOB)
        
        result = _parse_aamva_barcode(mock_payload)
        
        self.assertEqual(result["last_name"], "DOE")
        self.assertEqual(result["first_name"], "Unknown")
        self.assertEqual(result["birth_date"], "Unknown")

    def test_get_identaty_unsupported_type(self):
        """
        Test that requesting an unsupported ID type raises a ValueError.
        """
        with self.assertRaises(ValueError):
            get_identaty("passport")

    @patch('user_identaty_util._get_from_drivers_license')
    def test_get_identaty_routing(self, mock_get_license):
        """
        Test that the main routing function correctly calls the driver's license
        function when requested. The GUI function is mocked to prevent it from 
        hanging the test suite.
        """
        # Define what the mocked GUI function should return
        mock_get_license.return_value = {
            "first_name": "TestUser",
            "last_name": "Ontario",
            "birth_date": "1999-05-15"
        }
        
        result = get_identaty("drivers_licance")
        
        # Verify the routing function returned the data and called the correct sub-function
        self.assertEqual(result["first_name"], "TestUser")
        mock_get_license.assert_called_once()

if __name__ == '__main__':
    unittest.main()