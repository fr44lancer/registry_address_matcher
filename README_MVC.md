# Address Registry Matcher - MVC Architecture

This is a decomposed version of the original `address.py` file, restructured using Model-View-Controller (MVC) architecture while preserving all the exact logic and functionality.

## Architecture Overview

The application has been decomposed into the following components:

### 📁 Project Structure

```
registry_address_matcher/
├── app.py                      # Main application entry point
├── config.py                   # Configuration settings
├── requirements_mvc.txt        # Dependencies
├── models/                     # Model layer (Backend Logic)
│   ├── __init__.py
│   └── address_models.py       # Data models and business logic
├── views/                      # View layer (Frontend/UI)
│   ├── __init__.py
│   ├── ui_components.py        # UI components and layouts
│   └── visualizations.py      # Chart and graph components
├── controllers/                # Controller layer (Business Logic)
│   ├── __init__.py
│   └── matching_controller.py  # Main application controller
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── export_utils.py         # Export and utility functions
└── data/                       # Data files
    ├── spr.csv
    └── cadastre.csv
```

## 🔧 Components Breakdown

### Model Layer (`models/`)

**File: `address_models.py`**

Contains all the core business logic and data processing:

- `AddressNormalizer`: Text normalization and Armenian language handling
- `AdvancedAddressMatcher`: Core matching algorithms (exact and fuzzy)
- Data loading functions (`load_registry_data_from_csv`)
- Data preprocessing pipeline (`preprocess_registries`)
- Data quality analysis (`analyze_data_quality`)

**Key Features:**
- Complete Armenian street name normalization
- Multi-strategy fuzzy matching with progress tracking
- Exact matching with multiple approaches
- Comprehensive indexing for performance
- Data quality metrics calculation

### View Layer (`views/`)

**File: `ui_components.py`**

Contains all UI components and layouts:

- Page configuration and styling
- Sidebar configuration panel
- Data overview metrics display
- Matching controls and progress indicators
- Results summary and statistics
- Interactive match explorer
- Manual review interface
- Export options and recommendations

**File: `visualizations.py`**

Contains all chart and visualization components:

- Match quality analysis charts
- Data quality comparison dashboards
- Score distribution visualizations
- Performance timeline charts
- Street-level analysis charts
- Advanced matching analysis dashboards

### Controller Layer (`controllers/`)

**File: `matching_controller.py`**

Contains the main application logic and orchestration:

- `MatchingController`: Main controller class
- Session state management
- Data loading coordination
- Matching process orchestration
- Results handling and storage
- Export coordination
- Error handling and user feedback

**Key Features:**
- Complete matching process orchestration
- Progress tracking and user feedback
- Performance monitoring and warnings
- Result caching and session management
- Export functionality coordination

### Configuration (`config.py`)

Contains all application configuration:

- Database configuration
- Application settings
- Matching algorithm parameters
- Armenian language configuration
- Export settings

### Utilities (`utils/`)

**File: `export_utils.py`**

Contains export and utility functions:

- Comprehensive export package creation
- CSV/Excel export functions
- Summary report generation
- Quality report creation

## 🚀 Running the Application

1. **Install dependencies:**
```bash
pip install -r requirements_mvc.txt
```

2. **Test imports (recommended):**
```bash
python test_imports.py
```

3. **Run the application:**
```bash
# Option A: Using the run script (Recommended)
python run_app.py

# Option B: Direct Streamlit command
streamlit run app.py
```

4. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - The application will load with the main dashboard

### 🚨 Important Notes

- Always run from the project root directory
- Use `run_app.py` for the most reliable startup
- If you encounter import errors, check the SETUP_GUIDE.md

## 📊 Preserved Functionality

The MVC version preserves **100% of the original functionality**:

### ✅ Core Features Preserved:
- Complete Armenian address normalization
- Exact matching (full address + search key)
- Advanced fuzzy matching with multiple strategies
- Real-time progress tracking with detailed statistics
- Interactive results analysis
- Manual review interface
- Comprehensive export options
- Data quality analysis
- Performance monitoring

### ✅ UI Features Preserved:
- 4-tab interface (Data Overview, Matching Process, Results Analysis, Quality Report)
- Sidebar configuration panel
- Real-time progress bars and metrics
- Interactive filtering and pagination
- Color-coded match quality indicators
- Score distribution analysis
- Match type breakdown
- Export functionality

### ✅ Technical Features Preserved:
- Session state management
- Stop/reset functionality
- Performance warnings and recommendations
- Error handling and logging
- Data validation
- Caching mechanisms

## 🔍 Key Improvements

While preserving all functionality, the MVC structure provides:

1. **Better Code Organization**: Clear separation of concerns
2. **Maintainability**: Easier to modify and extend
3. **Testability**: Components can be tested independently
4. **Reusability**: Models and views can be reused
5. **Scalability**: Easier to add new features

## 🎯 Usage

The application works exactly like the original `address.py`:

1. **Data Overview Tab**: Load and analyze registry data
2. **Matching Process Tab**: Configure and run matching algorithms
3. **Results Analysis Tab**: Analyze and explore matching results
4. **Quality Report Tab**: Generate reports and export results

## 📈 Performance

The MVC version maintains the same performance characteristics:

- Chunk-based processing for large datasets
- Progress tracking with ETA calculation
- Memory-efficient operations
- Configurable processing limits

## 🛠️ Configuration

All configuration is centralized in `config.py`:

- Database settings
- Processing parameters
- UI configuration
- Export settings
- Language-specific settings

## 🔄 Migration from Original

To migrate from the original `address.py`:

1. Use `app.py` instead of `address.py`
2. All functionality remains the same
3. Configuration is now in `config.py`
4. Dependencies are in `requirements_mvc.txt`

The MVC version is a drop-in replacement with improved architecture while preserving all original functionality and user experience.