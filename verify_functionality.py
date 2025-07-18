#!/usr/bin/env python3
"""
Verification script to ensure MVC version preserves all functionality from original address.py
"""

import sys
import os

def verify_file_structure():
    """Verify all MVC files exist"""
    print("🔍 Verifying MVC file structure...")
    
    required_files = [
        'app.py',
        'config.py',
        'models/__init__.py',
        'models/address_models.py',
        'views/__init__.py',
        'views/ui_components.py',
        'views/visualizations.py',
        'controllers/__init__.py',
        'controllers/matching_controller.py',
        'utils/__init__.py',
        'utils/export_utils.py',
        'requirements_mvc.txt',
        'README_MVC.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All MVC files exist")
        return True

def verify_core_classes():
    """Verify core classes have all required methods"""
    print("\n🔍 Verifying core classes...")
    
    # Check AddressNormalizer
    with open('models/address_models.py', 'r') as f:
        content = f.read()
    
    required_methods = [
        'class AddressNormalizer',
        'def __init__(self)',
        'def _norm(self, text)',
        'def normalize(self, text)'
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ AddressNormalizer missing methods: {missing_methods}")
        return False
    else:
        print("✅ AddressNormalizer has all required methods")
    
    # Check AdvancedAddressMatcher
    required_matcher_methods = [
        'class AdvancedAddressMatcher',
        'def __init__(self, spr_df, cad_df, max_records=None)',
        'def find_exact_matches(self)',
        'def find_fuzzy_matches(self, threshold=85, chunk_size=1000, exclude_spr_ids=None)',
        'def _find_best_fuzzy_match(self, spr_row, threshold)',
        'def _create_match_record(self, spr_row, cad_row, score, match_type, candidates_count=1)'
    ]
    
    missing_matcher_methods = []
    for method in required_matcher_methods:
        if method not in content:
            missing_matcher_methods.append(method)
    
    if missing_matcher_methods:
        print(f"❌ AdvancedAddressMatcher missing methods: {missing_matcher_methods}")
        return False
    else:
        print("✅ AdvancedAddressMatcher has all required methods")
    
    return True

def verify_ui_components():
    """Verify UI components exist"""
    print("\n🔍 Verifying UI components...")
    
    with open('views/ui_components.py', 'r') as f:
        content = f.read()
    
    required_ui_functions = [
        'def configure_page()',
        'def apply_custom_css()',
        'def render_main_header()',
        'def render_sidebar_config()',
        'def render_data_overview_metrics',
        'def render_matching_controls()',
        'def render_interactive_match_explorer'
    ]
    
    missing_ui_functions = []
    for func in required_ui_functions:
        if func not in content:
            missing_ui_functions.append(func)
    
    if missing_ui_functions:
        print(f"❌ UI components missing functions: {missing_ui_functions}")
        return False
    else:
        print("✅ All UI components exist")
    
    return True

def verify_controller():
    """Verify controller functionality"""
    print("\n🔍 Verifying controller...")
    
    with open('controllers/matching_controller.py', 'r') as f:
        content = f.read()
    
    required_controller_methods = [
        'class MatchingController',
        'def initialize_session_state(self)',
        'def load_data(self, config',
        'def handle_matching_process(self, config',
        'def handle_stop_request(self)',
        'def handle_reset_request(self)'
    ]
    
    missing_controller_methods = []
    for method in required_controller_methods:
        if method not in content:
            missing_controller_methods.append(method)
    
    if missing_controller_methods:
        print(f"❌ Controller missing methods: {missing_controller_methods}")
        return False
    else:
        print("✅ Controller has all required methods")
    
    return True

def verify_exports():
    """Verify export functionality"""
    print("\n🔍 Verifying export functionality...")
    
    with open('utils/export_utils.py', 'r') as f:
        content = f.read()
    
    required_export_functions = [
        'def create_export_package',
        'def export_matches_to_csv',
        'def create_summary_report',
        'def create_quality_report'
    ]
    
    missing_export_functions = []
    for func in required_export_functions:
        if func not in content:
            missing_export_functions.append(func)
    
    if missing_export_functions:
        print(f"❌ Export functions missing: {missing_export_functions}")
        return False
    else:
        print("✅ All export functions exist")
    
    return True

def verify_configuration():
    """Verify configuration completeness"""
    print("\n🔍 Verifying configuration...")
    
    with open('config.py', 'r') as f:
        content = f.read()
    
    required_config_items = [
        'def setup_logging()',
        'def get_database_config()',
        'def get_app_config()',
        'def get_matching_config()',
        'ARMENIAN_CONFIG',
        'EXPORT_CONFIG'
    ]
    
    missing_config_items = []
    for item in required_config_items:
        if item not in content:
            missing_config_items.append(item)
    
    if missing_config_items:
        print(f"❌ Configuration missing items: {missing_config_items}")
        return False
    else:
        print("✅ All configuration items exist")
    
    return True

def verify_armenian_normalization():
    """Verify Armenian normalization logic is preserved"""
    print("\n🔍 Verifying Armenian normalization logic...")
    
    with open('models/address_models.py', 'r') as f:
        content = f.read()
    
    # Check for Armenian street name mappings
    armenian_checks = [
        'Ֆրունզեի',
        'Լենինգրադյան',
        'Կալինինի',
        'old_to_new_map',
        'armenian_suffixes'
    ]
    
    missing_armenian = []
    for check in armenian_checks:
        if check not in content:
            missing_armenian.append(check)
    
    if missing_armenian:
        print(f"❌ Armenian normalization missing: {missing_armenian}")
        return False
    else:
        print("✅ Armenian normalization logic preserved")
    
    return True

def verify_app_structure():
    """Verify main app structure"""
    print("\n🔍 Verifying main app structure...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    app_checks = [
        'def main()',
        'tab1, tab2, tab3, tab4 = st.tabs(',
        'Data Overview',
        'Matching Process',
        'Results Analysis',
        'Quality Report',
        'MatchingController'
    ]
    
    missing_app_items = []
    for check in app_checks:
        if check not in content:
            missing_app_items.append(check)
    
    if missing_app_items:
        print(f"❌ App structure missing: {missing_app_items}")
        return False
    else:
        print("✅ App structure preserved")
    
    return True

def main():
    """Main verification function"""
    print("🚀 Starting MVC functionality verification...\n")
    
    tests = [
        verify_file_structure,
        verify_core_classes,
        verify_ui_components,
        verify_controller,
        verify_exports,
        verify_configuration,
        verify_armenian_normalization,
        verify_app_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print(f"\n📊 Verification Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All verification tests passed! MVC version preserves all functionality.")
        return True
    else:
        print(f"\n⚠️  {failed} verification tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)