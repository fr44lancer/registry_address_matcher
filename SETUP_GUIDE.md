# 🚀 Setup Guide for Address Registry Matcher (MVC Version)

This guide will help you set up and run the MVC version of the Address Registry Matcher application.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements_mvc.txt
```

### 2. Verify Installation

Test that all imports work correctly:

```bash
python test_imports.py
```

You should see output like:
```
Testing imports...
✅ Models imported successfully
✅ Views imported successfully
✅ Controllers imported successfully
✅ Utils imported successfully
✅ AddressNormalizer working: Ֆրունզեի -> Լ ՄԱԴՈՅԱՆ
✅ MatchingController initialized successfully

🎉 All imports and basic functionality tests passed!
```

### 3. Run the Application

**Option A: Using the run script (Recommended)**
```bash
python run_app.py
```

**Option B: Direct Streamlit command**
```bash
`streamlit run app.py`
```

**Option C: Using Python module**
```bash
python -m streamlit run app.py
```

## 🛠️ Troubleshooting

### Import Errors

If you encounter import errors like:
```
ImportError: attempted relative import beyond top-level package
```

**Solution:** Always run the application from the project root directory:
```bash
cd /var/www/html/registry_address_matcher
python run_app.py
```

### Missing Dependencies

If you get errors about missing packages:
```bash
pip install pandas>=1.5.0 streamlit>=1.25.0 plotly>=5.0.0 rapidfuzz>=2.0.0
```

### Path Issues

If you encounter path-related issues:
1. Make sure you're in the correct directory
2. Use the provided `run_app.py` script
3. Check that all Python files are in the correct locations

## 📁 Project Structure

```
registry_address_matcher/
├── app.py                      # Main application entry point
├── run_app.py                  # Application runner script
├── test_imports.py             # Import verification script
├── verify_functionality.py    # Functionality verification
├── config.py                   # Configuration settings
├── requirements_mvc.txt        # Dependencies
├── models/                     # Model layer (Backend Logic)
│   ├── __init__.py
│   └── address_models.py
├── views/                      # View layer (Frontend/UI)
│   ├── __init__.py
│   ├── ui_components.py
│   └── visualizations.py
├── controllers/                # Controller layer (Business Logic)
│   ├── __init__.py
│   └── matching_controller.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── export_utils.py
└── data/                       # Data files
    ├── spr.csv
    └── cadastre.csv
```

## 🎯 Usage Instructions

1. **Start the application:**
   ```bash
   python run_app.py
   ```

2. **Access the web interface:**
   - Open your browser to `http://localhost:8501`
   - The application will load with the main dashboard

3. **Use the 4-tab interface:**
   - **📊 Data Overview**: Load and analyze registry data
   - **🔍 Matching Process**: Configure and run matching algorithms
   - **📈 Results Analysis**: Analyze and explore matching results
   - **📋 Quality Report**: Generate reports and export results

## 🔍 Testing

Run the functionality verification:
```bash
python verify_functionality.py
```

This will verify that all components are working correctly and that the MVC version preserves 100% of the original functionality.

## 🚨 Common Issues and Solutions

### 1. "No module named 'models'" Error

**Problem:** Python can't find the custom modules.

**Solution:** 
- Make sure you're running from the project root directory
- Use the `run_app.py` script which sets up the path correctly

### 2. "ModuleNotFoundError: No module named 'pandas'"

**Problem:** Missing required dependencies.

**Solution:**
```bash
pip install -r requirements_mvc.txt
```

### 3. Streamlit not starting

**Problem:** Streamlit command not found.

**Solution:**
```bash
pip install streamlit
# or
pip install -r requirements_mvc.txt
```

### 4. Data files not found

**Problem:** CSV files missing from data directory.

**Solution:**
- Ensure `data/spr.csv` and `data/cadastre.csv` exist
- Check file permissions

## 📊 Features

The MVC version includes all original features:

✅ **Complete Armenian address normalization** (206 street mappings)
✅ **Exact matching** (full address + search key strategies)  
✅ **Advanced fuzzy matching** with 4 different strategies
✅ **Real-time progress tracking** with detailed statistics
✅ **Interactive results analysis** with filtering and exploration
✅ **Manual review interface** for quality control
✅ **Comprehensive export options** (CSV, Excel, ZIP packages)
✅ **Data quality analysis** and reporting
✅ **Performance monitoring** and optimization

## 🎉 Success!

If everything is working correctly, you should see:
- The Streamlit app running at `http://localhost:8501`
- All four tabs accessible and functional
- Data loading and processing capabilities
- Matching algorithms working with progress tracking
- Export functionality operational

## 🆘 Need Help?

If you encounter any issues:
1. Check that all dependencies are installed
2. Verify you're running from the correct directory
3. Run `python test_imports.py` to diagnose import issues
4. Run `python verify_functionality.py` to check functionality
5. Check the console output for specific error messages

The MVC version is designed to be a drop-in replacement for the original `address.py` with improved architecture while maintaining 100% functional compatibility.