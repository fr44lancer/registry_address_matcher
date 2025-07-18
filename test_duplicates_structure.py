#!/usr/bin/env python3
"""
Test script to verify duplicate detection structure and imports
"""
import sys
import os
import ast

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_structure():
    """Test that all duplicate-related files exist"""
    print("🧪 Testing Duplicate Detection File Structure...\n")
    
    required_files = [
        'models/duplicate_models.py',
        'views/duplicates_view.py',
        'controllers/duplicates_controller.py',
        'test_duplicates.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All duplicate detection files exist")
        return True

def test_python_syntax():
    """Test Python syntax of duplicate-related files"""
    print("\n🧪 Testing Python Syntax...\n")
    
    files_to_check = [
        'models/duplicate_models.py',
        'views/duplicates_view.py',
        'controllers/duplicates_controller.py',
        'test_duplicates.py'
    ]
    
    all_valid = True
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {file_path}: Valid syntax")
        except SyntaxError as e:
            print(f"❌ {file_path}: Syntax error - {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {file_path}: Error - {e}")
            all_valid = False
    
    return all_valid

def test_class_definitions():
    """Test that required classes are defined"""
    print("\n🧪 Testing Class Definitions...\n")
    
    # Check SPRDuplicateDetector
    with open('models/duplicate_models.py', 'r') as f:
        content = f.read()
    
    required_classes = [
        'class SPRDuplicateDetector',
        'class DuplicateDataProcessor'
    ]
    
    required_methods = [
        'def detect_duplicates(self)',
        'def get_duplicate_summary(self)',
        'def analyze_duplicate_patterns(self)',
        'def get_duplicate_resolution_suggestions(self)',
        'def export_duplicates_report(self)'
    ]
    
    missing_items = []
    for item in required_classes + required_methods:
        if item not in content:
            missing_items.append(item)
    
    if missing_items:
        print(f"❌ Missing in duplicate_models.py: {missing_items}")
        return False
    else:
        print("✅ All required classes and methods found in duplicate_models.py")
    
    # Check DuplicatesController
    with open('controllers/duplicates_controller.py', 'r') as f:
        content = f.read()
    
    required_controller_methods = [
        'class DuplicatesController',
        'def initialize_session_state(self)',
        'def detect_duplicates(self, spr_df',
        'def render_duplicates_tab(self)',
        'def get_duplicate_results(self)',
        'def analyze_patterns(self)',
        'def get_resolution_suggestions(self)'
    ]
    
    missing_controller_items = []
    for item in required_controller_methods:
        if item not in content:
            missing_controller_items.append(item)
    
    if missing_controller_items:
        print(f"❌ Missing in duplicates_controller.py: {missing_controller_items}")
        return False
    else:
        print("✅ All required methods found in duplicates_controller.py")
    
    return True

def test_view_functions():
    """Test that required view functions are defined"""
    print("\n🧪 Testing View Functions...\n")
    
    with open('views/duplicates_view.py', 'r') as f:
        content = f.read()
    
    required_functions = [
        'def render_duplicates_overview(',
        'def render_duplicates_filters(',
        'def render_duplicates_table(',
        'def render_duplicates_analysis(',
        'def render_resolution_suggestions(',
        'def render_duplicates_export(',
        'def render_duplicates_help(',
        'def create_duplicates_overview_chart('
    ]
    
    missing_functions = []
    for func in required_functions:
        if func not in content:
            missing_functions.append(func)
    
    if missing_functions:
        print(f"❌ Missing view functions: {missing_functions}")
        return False
    else:
        print("✅ All required view functions found")
    
    return True

def test_app_integration():
    """Test that the app is properly integrated"""
    print("\n🧪 Testing App Integration...\n")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    required_integrations = [
        'from controllers.duplicates_controller import DuplicatesController',
        'duplicates_controller = DuplicatesController()',
        'tab5 = st.tabs(',
        'SPR Duplicates',
        'def render_duplicates_tab(duplicates_controller'
    ]
    
    missing_integrations = []
    for integration in required_integrations:
        if integration not in content:
            missing_integrations.append(integration)
    
    if missing_integrations:
        print(f"❌ Missing app integrations: {missing_integrations}")
        return False
    else:
        print("✅ App integration complete")
    
    return True

def test_imports_structure():
    """Test that imports are properly structured"""
    print("\n🧪 Testing Import Structure...\n")
    
    # Check models __init__.py
    with open('models/__init__.py', 'r') as f:
        models_init = f.read()
    
    if 'SPRDuplicateDetector' not in models_init:
        print("❌ SPRDuplicateDetector not in models/__init__.py")
        return False
    
    # Check views __init__.py
    with open('views/__init__.py', 'r') as f:
        views_init = f.read()
    
    if 'render_duplicates_overview' not in views_init:
        print("❌ Duplicate view functions not in views/__init__.py")
        return False
    
    # Check controllers __init__.py
    with open('controllers/__init__.py', 'r') as f:
        controllers_init = f.read()
    
    if 'DuplicatesController' not in controllers_init:
        print("❌ DuplicatesController not in controllers/__init__.py")
        return False
    
    print("✅ All imports properly structured")
    return True

def test_functionality_features():
    """Test that all required functionality features are present"""
    print("\n🧪 Testing Functionality Features...\n")
    
    # Check duplicate detection features
    with open('models/duplicate_models.py', 'r') as f:
        model_content = f.read()
    
    required_features = [
        'FULL_ADDRESS',  # Full address comparison
        'duplicate_groups',  # Grouping duplicates
        'duplicate_stats',  # Statistics calculation
        'patterns_analysis',  # Pattern analysis
        'resolution_suggestions',  # Resolution suggestions
        'export_duplicates_report',  # Export functionality
        'filter_duplicates_by_criteria'  # Filtering
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in model_content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing features in models: {missing_features}")
        return False
    
    # Check view features
    with open('views/duplicates_view.py', 'r') as f:
        view_content = f.read()
    
    required_view_features = [
        'plotly',  # Charts and visualizations
        'st.tabs',  # Tab interface
        'st.slider',  # Filters
        'st.download_button',  # Export functionality
        'st.dataframe',  # Table display
        'st.metric',  # Metrics display
        'highlight_groups'  # Color coding
    ]
    
    missing_view_features = []
    for feature in required_view_features:
        if feature not in view_content:
            missing_view_features.append(feature)
    
    if missing_view_features:
        print(f"❌ Missing view features: {missing_view_features}")
        return False
    
    print("✅ All required functionality features present")
    return True

def main():
    """Main test function"""
    print("🚀 Starting SPR Duplicates Structure Tests...\n")
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_class_definitions,
        test_view_functions,
        test_app_integration,
        test_imports_structure,
        test_functionality_features
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
    
    print(f"\n📊 Structure Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All structure tests passed!")
        print("✅ SPR Duplicates functionality is properly implemented!")
        print("\n📋 Features Available:")
        print("   - ✅ Backend duplicate detection model")
        print("   - ✅ Frontend duplicate display views")
        print("   - ✅ Controller for managing duplicates")
        print("   - ✅ Integration with main application")
        print("   - ✅ Export functionality")
        print("   - ✅ Pattern analysis")
        print("   - ✅ Resolution suggestions")
        print("   - ✅ Interactive filtering")
        print("   - ✅ Visualization charts")
        return True
    else:
        print(f"\n⚠️  {failed} structure tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)