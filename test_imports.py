#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    try:
        print("Testing imports...")
        
        # Test models
        from models.address_models import AddressNormalizer, AdvancedAddressMatcher
        print("✅ Models imported successfully")
        
        # Test views
        from views.ui_components import configure_page, render_main_header
        print("✅ Views imported successfully")
        
        # Test controllers
        from controllers.matching_controller import MatchingController
        print("✅ Controllers imported successfully")
        
        # Test utils
        from utils.export_utils import create_export_package
        print("✅ Utils imported successfully")
        
        # Test basic functionality
        normalizer = AddressNormalizer()
        test_text = "Ֆրունզեի"
        normalized = normalizer.normalize(test_text)
        print(f"✅ AddressNormalizer working: {test_text} -> {normalized}")
        
        controller = MatchingController()
        print("✅ MatchingController initialized successfully")
        
        print("\n🎉 All imports and basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)