# RALF - README



## 📋 Prerequisites

- Python 3.8 or higher (3.10)
- pip (Python package installer)
- Git

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ralf
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## 🧪 Running Tests

### Basic Test Execution

Run all tests:
```bash
python -m pytest tests/
```

Run tests with verbose output:
```bash
python -m pytest tests/ -v
```

Run tests with detailed error information:
```bash
python -m pytest tests/ --tb=long
```

### Specific Test Categories

Run only initialization tests:
```bash
python -m pytest tests/test_ralf.py::TestRalfInitialization -v
```

Run only data processing tests:
```bash
python -m pytest tests/test_ralf.py::TestDataLoadingAndProcessing -v
```

Run only model configuration tests:
```bash
python -m pytest tests/test_ralf.py::TestModelConfiguration -v
```

Run only state management tests:
```bash
python -m pytest tests/test_ralf.py::TestStateManagement -v
```

### Parameterized Tests

The test suite includes parameterized tests for various scenarios:

- **API Key Management**: Tests different combinations of API keys
- **Model Loading**: Tests with different transformer models (BERT, RoBERTa)
- **Data Processing**: Tests with various dataset sizes and configurations
- **Trainer Initialization**: Tests with different output paths

### Test Coverage

The test suite covers:
- ✅ Initialization and default attributes
- ✅ API key management
- ✅ Data loading and processing
- ✅ Model configuration with LoRA
- ✅ State saving and loading
- ✅ Trainer initialization
- ✅ Error handling and edge cases
- ✅ Hardware detection
- ✅ Integration workflows

## Code Coverage

### Installing Coverage Tools

Install pytest-cov for coverage analysis:
```bash
pip install pytest-cov
```

### Running Coverage Analysis

Generate coverage report:
```bash
python -m pytest tests/ --cov=ralf --cov-report=term-missing
```

Generate HTML coverage report:
```bash
python -m pytest tests/ --cov=ralf --cov-report=html
```

Generate XML coverage report (for CI/CD):
```bash
python -m pytest tests/ --cov=ralf --cov-report=xml
```

Generate multiple report formats:
```bash
python -m pytest tests/ --cov=ralf --cov-report=term-missing --cov-report=html --cov-report=xml
```


### Viewing Coverage Reports

After generating HTML coverage:
```bash
# Open the HTML report in your browser
open htmlcov/index.html
```

## Development Workflow

### Running Tests During Development

1. **Quick test run:**
   ```bash
   python -m pytest tests/ -x  # Stop on first failure
   ```

2. **Run tests with coverage:**
   ```bash
   python -m pytest tests/ --cov=ralf --cov-report=term-missing -v
   ```

3. **Run specific failing tests:**
   ```bash
   python -m pytest tests/test_ralf.py::TestSpecificClass::test_specific_method -v
   ```

### Continuous Integration

For CI/CD pipelines, use:
```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=ralf --cov-report=xml --cov-report=term-missing

# Optional: fail if coverage is below threshold
python -m pytest tests/ --cov=ralf --cov-fail-under=80
```

## 📁 Project Structure

```
ralf/
├── ralf/
│   ├── __init__.py
│   └── ralf.py          # Main Ralf class implementation
├── tests/
│   └── test_ralf.py     # Comprehensive test suite
├── requirements.txt     # Python dependencies
├── setup.py            # Package configuration
└── README.md           # This file
```

## 🧪 Test Suite Overview

The test suite is organized into logical test classes:

### TestRalfInitialization
- Tests default attribute initialization
- Hardware detection functionality

### TestAPIKeyManagement
- Parameterized tests for API key combinations
- Tests setting and retrieving API keys

### TestDataLoadingAndProcessing
- Data loading with various configurations
- Column renaming functionality
- Multi-class classification support
- Large dataset handling
- Error cases and edge conditions

### TestModelConfiguration
- Model loading with different architectures
- LoRA configuration and setup
- Error handling for missing data

### TestStateManagement
- Save/load functionality using pickle
- File creation and directory handling
- State restoration verification

### TestTrainerInitialization
- Trainer setup with custom configurations
- Path handling and validation
- Error cases for missing models

### TestErrorHandling
- Edge cases and error conditions
- Single label datasets
- Very small datasets

### TestHardwareDetection
- GPU and RAM detection
- Hardware attribute validation

### TestIntegration
- Complete workflow testing
- End-to-end functionality verification

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Make sure you're in the project directory
   cd ralf
   
   # Install in development mode
   pip install -e .
   ```

2. **Missing Dependencies:**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   pip install pytest-cov
   ```

3. **Test Failures:**
   ```bash
   # Run with verbose output to see details
   python -m pytest tests/ -v --tb=long
   ```

4. **Coverage Issues:**
   ```bash
   # Check if pytest-cov is installed
   pip install pytest-cov
   
   # Run coverage with source specification
   python -m pytest tests/ --cov=ralf --cov-report=term-missing
   ```

### Environment Setup

For a clean development environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest-cov

# Install package in development mode
pip install -e .
```

## Performance Notes

- Tests use small datasets for quick execution
- Model downloads are cached between test runs
- GPU tests are skipped if CUDA is not available
- Memory usage is optimized for CI/CD environments

## Contributing

When contributing to the project:

1. **Write tests** for new functionality
2. **Run the full test suite** before submitting
3. **Maintain coverage** above 80%
4. **Follow the existing test patterns** and structure


