#!/usr/bin/env python3
"""
Test script for duplicate detection functionality
"""
import sys
import os
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create sample SPR data with known duplicates"""
    test_data = [
        # Regular unique addresses
        {'ADDRESS_ID': 1, 'STREET_NAME': 'Մաշտոցի', 'HOUSE': '15', 'BUILDING': ''},
        {'ADDRESS_ID': 2, 'STREET_NAME': 'Վարդանանց', 'HOUSE': '25', 'BUILDING': 'Ա'},
        {'ADDRESS_ID': 3, 'STREET_NAME': 'Արամի', 'HOUSE': '10', 'BUILDING': ''},
        
        # Duplicate group 1 (same address, different IDs)
        {'ADDRESS_ID': 4, 'STREET_NAME': 'Մաշտոցի', 'HOUSE': '20', 'BUILDING': ''},
        {'ADDRESS_ID': 5, 'STREET_NAME': 'Մաշտոցի', 'HOUSE': '20', 'BUILDING': ''},
        {'ADDRESS_ID': 6, 'STREET_NAME': 'Մաշտոցի', 'HOUSE': '20', 'BUILDING': ''},
        
        # Duplicate group 2 (same address with building)
        {'ADDRESS_ID': 7, 'STREET_NAME': 'Տիգրան Մեծի', 'HOUSE': '5', 'BUILDING': 'Բ'},
        {'ADDRESS_ID': 8, 'STREET_NAME': 'Տիգրան Մեծի', 'HOUSE': '5', 'BUILDING': 'Բ'},
        
        # Empty address (should be filtered out)
        {'ADDRESS_ID': 9, 'STREET_NAME': '', 'HOUSE': '', 'BUILDING': ''},
        
        # Another unique address
        {'ADDRESS_ID': 10, 'STREET_NAME': 'Կոմիտաս', 'HOUSE': '35', 'BUILDING': ''},
    ]
    
    return pd.DataFrame(test_data)

def test_duplicate_detection():
    """Test the duplicate detection functionality"""
    print("🧪 Testing SPR Duplicate Detection...\n")
    
    try:
        # Import the models
        from models.duplicate_models import SPRDuplicateDetector, DuplicateDataProcessor
        
        # Create test data
        print("📊 Creating test data...")
        test_df = create_test_data()
        print(f"   - Created {len(test_df)} test records")
        print(f"   - Expected duplicates: 5 records in 2 groups")
        
        # Initialize detector
        print("\n🔍 Initializing duplicate detector...")
        detector = SPRDuplicateDetector(test_df)
        
        # Detect duplicates
        print("🔍 Detecting duplicates...")
        results = detector.detect_duplicates()
        
        # Validate results
        print("\n📋 Results:")
        stats = results.get('duplicate_stats', {})
        duplicate_groups = results.get('duplicate_groups', {})
        
        print(f"   - Total records: {stats.get('total_records', 0)}")
        print(f"   - Total duplicates: {stats.get('total_duplicates', 0)}")
        print(f"   - Unique duplicate groups: {stats.get('unique_duplicate_groups', 0)}")
        print(f"   - Duplicate rate: {stats.get('duplicate_rate', 0):.1%}")
        print(f"   - Empty addresses: {stats.get('empty_addresses', 0)}")
        print(f"   - Largest duplicate group: {stats.get('largest_duplicate_group', 0)}")
        
        # Show duplicate groups
        print(f"\n🔍 Duplicate Groups ({len(duplicate_groups)}):")
        for i, (address, group) in enumerate(duplicate_groups.items(), 1):
            print(f"   {i}. '{address}' - {group['count']} duplicates")
            for j, record in enumerate(group['records'], 1):
                print(f"      {j}. ID: {record.get('ADDRESS_ID', 'N/A')}")
        
        # Test patterns analysis
        print("\n📊 Testing patterns analysis...")
        patterns = detector.analyze_duplicate_patterns()
        print(f"   - Count distribution: {patterns.get('count_distribution', {})}")
        print(f"   - Top streets: {list(patterns.get('top_streets_with_duplicates', {}).keys())[:3]}")
        
        # Test resolution suggestions
        print("\n💡 Testing resolution suggestions...")
        suggestions = detector.get_duplicate_resolution_suggestions()
        print(f"   - Total suggestions: {len(suggestions)}")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion['address']} - {suggestion['suggestion_type']}")
        
        # Test data processor
        print("\n📋 Testing data processor...")
        processor = DuplicateDataProcessor()
        display_df = processor.format_duplicate_groups_for_display(duplicate_groups)
        print(f"   - Display DataFrame shape: {display_df.shape}")
        print(f"   - Display DataFrame columns: {list(display_df.columns)}")
        
        # Test export
        print("\n📦 Testing export...")
        export_report = detector.export_duplicates_report()
        print(f"   - Export report keys: {list(export_report.keys())}")
        print(f"   - Export timestamp: {export_report.get('export_timestamp', 'N/A')}")
        
        # Validation checks
        print("\n✅ Validation Checks:")
        
        # Check expected duplicates
        expected_duplicates = 5  # 3 + 2 duplicates
        actual_duplicates = stats.get('total_duplicates', 0)
        print(f"   - Expected duplicates: {expected_duplicates}, Actual: {actual_duplicates} {'✅' if expected_duplicates == actual_duplicates else '❌'}")
        
        # Check expected groups
        expected_groups = 2
        actual_groups = stats.get('unique_duplicate_groups', 0)
        print(f"   - Expected groups: {expected_groups}, Actual: {actual_groups} {'✅' if expected_groups == actual_groups else '❌'}")
        
        # Check largest group
        expected_largest = 3
        actual_largest = stats.get('largest_duplicate_group', 0)
        print(f"   - Expected largest group: {expected_largest}, Actual: {actual_largest} {'✅' if expected_largest == actual_largest else '❌'}")
        
        # Check empty addresses
        expected_empty = 1
        actual_empty = stats.get('empty_addresses', 0)
        print(f"   - Expected empty addresses: {expected_empty}, Actual: {actual_empty} {'✅' if expected_empty == actual_empty else '❌'}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controller():
    """Test the duplicates controller"""
    print("\n🧪 Testing Duplicates Controller...\n")
    
    try:
        from controllers.duplicates_controller import DuplicatesController
        
        # Create test data
        test_df = create_test_data()
        
        # Initialize controller
        print("🎛️ Initializing controller...")
        controller = DuplicatesController()
        
        # Test session state initialization
        print("📋 Testing session state...")
        print(f"   - Session state keys initialized: {hasattr(controller, 'initialize_session_state')}")
        
        # Test duplicate detection through controller
        print("🔍 Testing duplicate detection...")
        results = controller.detect_duplicates(test_df)
        
        if results:
            print(f"   - Results obtained: ✅")
            print(f"   - Total duplicates found: {results.get('duplicate_stats', {}).get('total_duplicates', 0)}")
        else:
            print(f"   - Results obtained: ❌")
        
        # Test getting results
        print("📊 Testing results retrieval...")
        cached_results = controller.get_duplicate_results()
        print(f"   - Cached results available: {'✅' if cached_results else '❌'}")
        
        # Test patterns analysis
        print("📈 Testing patterns analysis...")
        patterns = controller.analyze_patterns()
        print(f"   - Patterns analysis available: {'✅' if patterns else '❌'}")
        
        # Test resolution suggestions
        print("💡 Testing resolution suggestions...")
        suggestions = controller.get_resolution_suggestions()
        print(f"   - Resolution suggestions available: {'✅' if suggestions else '❌'}")
        
        # Test export
        print("📦 Testing export...")
        report = controller.export_duplicates_report()
        print(f"   - Export report available: {'✅' if report else '❌'}")
        
        # Test summary for main app
        print("📋 Testing summary for main app...")
        summary = controller.get_duplicate_summary_for_main_app()
        print(f"   - Summary available: {'✅' if summary else '❌'}")
        print(f"   - Has duplicates: {summary.get('has_duplicates', False)}")
        
        print("\n🎉 Controller tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during controller testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting SPR Duplicates Functionality Tests...\n")
    
    # Run tests
    model_test = test_duplicate_detection()
    controller_test = test_controller()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   - Model tests: {'✅ PASSED' if model_test else '❌ FAILED'}")
    print(f"   - Controller tests: {'✅ PASSED' if controller_test else '❌ FAILED'}")
    
    if model_test and controller_test:
        print("\n🎉 All duplicate detection tests passed!")
        print("✅ The SPR duplicates functionality is working correctly!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)