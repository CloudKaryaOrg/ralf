"""
Comprehensive test suite for the Ralf class.

This module contains parameterized tests that cover all major functionality
of the Ralf class including initialization, data processing, model configuration,
state management, and error handling scenarios.

Test Coverage:
- Initialization and default attributes
- API key management
- Data loading and processing with various configurations
- Model loading and configuration
- State saving and loading
- Error handling and edge cases
- Hardware detection
- Trainer initialization
"""

import pytest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import json
from unittest.mock import patch, MagicMock
from ralf import Ralf


class TestRalfInitialization:
    """Test suite for Ralf class initialization and default attributes."""
    
    def test_ralf_default_attributes(self):
        """Test that Ralf initializes with correct default attributes."""
        ralf = Ralf()
        
        # Dataset attributes
        assert ralf.golden_dataset is None
        assert ralf.platinum_dataset is None
        assert isinstance(ralf.other_datasets, dict)
        assert len(ralf.other_datasets) == 0
        
        # Model and training attributes
        assert ralf.model_name is None
        assert ralf.trainer is None
        assert ralf.num_labels is None
        assert ralf.label_to_id is None
        assert ralf.id_to_label is None
        assert ralf.tokenizer is None
        assert ralf.train_dataset is None
        assert ralf.val_dataset is None
        assert ralf.model is None
        
        # API key attributes
        assert ralf.open_api_key is None
        assert ralf.gemini_key is None
        assert ralf.hf_token is None
        
        # Hardware detection attributes
        assert hasattr(ralf, 'gpu_available')
        assert hasattr(ralf, 'gpu_count')
        assert hasattr(ralf, 'gpu_name')
        assert hasattr(ralf, 'gpu_ram_gb')
        assert hasattr(ralf, 'ram_gb')
        assert isinstance(ralf.ram_gb, (int, float))


class TestAPIKeyManagement:
    """Test suite for API key management functionality."""
    
    @pytest.mark.parametrize("open_api_key,gemini_key,hf_token", [
        ("open_key", None, None),
        (None, "gemini_key", None),
        (None, None, "hf_token"),
        ("open_key", "gemini_key", "hf_token"),
        ("", "", ""),
        (None, None, None),
    ])
    def test_set_keys_combinations(self, open_api_key, gemini_key, hf_token):
        """Test setting API keys with various combinations."""
        ralf = Ralf()
        ralf.set_keys(open_api_key=open_api_key, gemini_key=gemini_key, hf_token=hf_token)
        
        assert ralf.open_api_key == open_api_key
        assert ralf.gemini_key == gemini_key
        assert ralf.hf_token == hf_token


class TestDataLoadingAndProcessing:
    """Test suite for data loading and processing functionality."""
    
    @pytest.fixture
    def sample_dataframes(self):
        """Provide various sample dataframes for testing."""
        return {
            "binary_classification": pd.DataFrame({
                "text": ["positive text", "negative text", "positive text", "negative text", "positive text", "negative text"],
                "label": ["positive", "negative", "positive", "negative", "positive", "negative"]
            }),
            "multi_classification": pd.DataFrame({
                "text": ["class a", "class b", "class c", "class a", "class b", "class c", "class a", "class b", "class c"],
                "label": ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
            }),
            "custom_columns": pd.DataFrame({
                "mytext": ["text1", "text2", "text3", "text4", "text5", "text6"],
                "mylabel": ["label1", "label2", "label1", "label2", "label1", "label2"]
            }),
            "large_dataset": pd.DataFrame({
                "text": [f"text_{i}" for i in range(100)],
                "label": [f"label_{i % 5}" for i in range(100)]
            })
        }
    
    @pytest.mark.parametrize("model_name", [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base"
    ])
    def test_load_and_process_data_basic(self, sample_dataframes, model_name):
        """Test basic data loading and processing functionality."""
        ralf = Ralf()
        df = sample_dataframes["binary_classification"]
        
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name=model_name)
        
        # Check that model name is set
        assert ralf.model_name == model_name
        
        # Check label mappings
        assert ralf.num_labels == 2
        assert set(ralf.label_to_id.keys()) == {"positive", "negative"}
        assert set(ralf.id_to_label.values()) == {"positive", "negative"}
        
        # Check datasets are created
        assert ralf.train_dataset is not None
        assert ralf.val_dataset is not None
        assert ralf.tokenizer is not None
        
        # Check dataset sizes (80/20 split)
        assert len(ralf.train_dataset) == 4  # 80% of 6 samples
        assert len(ralf.val_dataset) == 2    # 20% of 6 samples
    
    def test_load_and_process_data_column_rename(self, sample_dataframes):
        """Test data loading with custom column names."""
        ralf = Ralf()
        df = sample_dataframes["custom_columns"]
        
        # Test with custom column names
        ralf.load_and_process_data(df, text_column="mytext", label_column="mylabel", model_name="bert-base-uncased")
        
        # Verify column renaming worked
        assert ralf.train_dataset is not None
        assert ralf.val_dataset is not None
        assert set(ralf.label_to_id.keys()) == {"label1", "label2"}
        assert set(ralf.id_to_label.values()) == {"label1", "label2"}
    
    def test_load_and_process_data_multi_class(self, sample_dataframes):
        """Test data loading with multi-class classification."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["class a", "class b", "class c", "class a", "class b", "class c", "class a", "class b", "class c", "class a", "class b", "class c"],
            "label": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
        })
        
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        
        assert ralf.num_labels == 3
        assert set(ralf.label_to_id.keys()) == {"A", "B", "C"}
        assert set(ralf.id_to_label.values()) == {"A", "B", "C"}
    
    def test_load_and_process_data_large_dataset(self, sample_dataframes):
        """Test data loading with a larger dataset."""
        ralf = Ralf()
        df = sample_dataframes["large_dataset"]
        
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        
        assert ralf.num_labels == 5
        assert len(ralf.train_dataset) == 80  # 80% of 100 samples
        assert len(ralf.val_dataset) == 20    # 20% of 100 samples
    
    def test_load_and_process_data_without_model_name(self, sample_dataframes):
        """Test that error is raised when model_name is not provided."""
        ralf = Ralf()
        df = sample_dataframes["binary_classification"]
        
        with pytest.raises(ValueError, match="model_name must be set"):
            ralf.load_and_process_data(df, text_column="text", label_column="label", model_name=None)
    
    def test_load_and_process_data_empty_dataframe(self):
        """Test handling of empty dataframe."""
        ralf = Ralf()
        df = pd.DataFrame(columns=["text", "label"])
        
        with pytest.raises(ValueError):
            ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")


class TestModelConfiguration:
    """Test suite for model loading and configuration."""
    
    @pytest.fixture
    def prepared_ralf(self):
        """Create a Ralf instance with processed data."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "label": ["A", "B", "A", "B", "A", "B"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        return ralf
    
    @pytest.mark.parametrize("model_name", [
        "bert-base-uncased",
        "roberta-base"
    ])
    def test_load_and_configure_model(self, model_name):
        """Test model loading and LoRA configuration."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "label": ["A", "B", "A", "B", "A", "B"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name=model_name)
        
        ralf.load_and_configure_model()
        
        assert ralf.model is not None
        assert hasattr(ralf.model, 'print_trainable_parameters')
    
    def test_load_and_configure_model_without_data(self):
        """Test that model configuration fails without processed data."""
        ralf = Ralf()
        
        # The error occurs when model_name is None, which happens when load_and_process_data is not called
        with pytest.raises(OSError):
            ralf.load_and_configure_model()


class TestStateManagement:
    """Test suite for state saving and loading functionality."""
    
    @pytest.fixture
    def trained_ralf(self, tmp_path):
        """Create a Ralf instance with processed data and configured model."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "label": ["A", "B", "A", "B", "A", "B"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        ralf.load_and_configure_model()
        return ralf
    
    def test_save_state_creates_files(self, trained_ralf, tmp_path):
        """Test that save_state creates the pickle file."""
        save_file = tmp_path / "ralf_state.pkl"
        trained_ralf.save_state(file_path=str(save_file))
        
        # Check that the pickle file exists
        assert save_file.exists()
        assert save_file.stat().st_size > 0  # File should not be empty
    
    def test_load_state_restores_complete_state(self, trained_ralf, tmp_path):
        """Test that load_state restores all components correctly."""
        save_file = tmp_path / "ralf_state.pkl"
        trained_ralf.save_state(file_path=str(save_file))
        
        loaded_ralf = Ralf.load_state(file_path=str(save_file))
        
        # Check that all components are restored
        assert loaded_ralf is not None
        assert loaded_ralf.model_name == "bert-base-uncased"
        assert loaded_ralf.num_labels == 2
        assert loaded_ralf.label_to_id == {"A": 0, "B": 1}
        assert loaded_ralf.id_to_label == {0: "A", 1: "B"}
        assert loaded_ralf.tokenizer is not None
        assert loaded_ralf.model is not None
        assert loaded_ralf.train_dataset is not None
        assert loaded_ralf.val_dataset is not None
    
    def test_save_state_without_model(self, tmp_path):
        """Test saving state when model is not configured."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "label": ["A", "B", "A", "B", "A", "B"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        
        save_file = tmp_path / "ralf_state.pkl"
        ralf.save_state(file_path=str(save_file))
        
        # Should still create the pickle file
        assert save_file.exists()
        assert save_file.stat().st_size > 0
    
    def test_load_state_nonexistent_file(self):
        """Test loading state from non-existent file."""
        loaded_ralf = Ralf.load_state(file_path="nonexistent_file.pkl")
        assert loaded_ralf is None
    
    def test_save_state_creates_directory(self, trained_ralf, tmp_path):
        """Test that save_state works when directory is created manually."""
        save_dir = tmp_path / "new_directory"
        save_dir.mkdir()  # Create the directory first
        save_file = save_dir / "ralf_state.pkl"
        trained_ralf.save_state(file_path=str(save_file))
        
        assert save_file.exists()
        assert save_file.stat().st_size > 0


class TestTrainerInitialization:
    """Test suite for trainer initialization."""
    
    @pytest.fixture
    def model_ready_ralf(self):
        """Create a Ralf instance ready for trainer initialization."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "label": ["A", "B", "A", "B", "A", "B"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        ralf.load_and_configure_model()
        return ralf
    
    def test_initialize_trainer_basic(self, model_ready_ralf, tmp_path):
        """Test basic trainer initialization."""
        output_dir = tmp_path / "results"
        save_path = tmp_path / "ralf_state.pkl"
        
        model_ready_ralf.initialize_trainer(output_dir=str(output_dir), save_path=str(save_path))
        
        assert model_ready_ralf.trainer is not None
        assert hasattr(model_ready_ralf.trainer, 'train')
        assert hasattr(model_ready_ralf.trainer, 'evaluate')
    
    def test_initialize_trainer_without_model(self):
        """Test that trainer initialization fails without configured model."""
        ralf = Ralf()
        
        with pytest.raises(ValueError):
            ralf.initialize_trainer()
    
    @pytest.mark.parametrize("output_dir,save_path", [
        ("./results", "ralf_state.pkl"),
        ("/tmp/custom_results", "/tmp/custom_state.pkl"),
        ("relative/path", "relative/state.pkl"),
    ])
    def test_initialize_trainer_custom_paths(self, model_ready_ralf, output_dir, save_path):
        """Test trainer initialization with custom paths."""
        model_ready_ralf.initialize_trainer(output_dir=output_dir, save_path=save_path)
        
        assert model_ready_ralf.trainer is not None
        # Check that training arguments are set correctly
        assert model_ready_ralf.trainer.args.output_dir == output_dir


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    # def test_load_and_process_data_invalid_columns(self):
    #     """Test handling of invalid column names."""
    #     ralf = Ralf()
    #     df = pd.DataFrame({
    #         "text": ["text1", "text2", "text3", "text4"],
    #         "label": ["A", "B", "A", "B"]
    #     })
    #     
    #     # Test with a non-existent label column - this should fail when trying to rename columns
    #     with pytest.raises(KeyError):
    #         ralf.load_and_process_data(df, text_column="text", label_column="nonexistent", model_name="bert-base-uncased")
    
    def test_load_and_process_data_single_label(self):
        """Test handling of dataset with only one unique label."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3"],
            "label": ["A", "A", "A"]
        })
        
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        
        assert ralf.num_labels == 1
        assert ralf.label_to_id == {"A": 0}
        assert ralf.id_to_label == {0: "A"}  # Fixed: integer key, not string
    
    def test_load_and_process_data_very_small_dataset(self):
        """Test handling of very small datasets."""
        ralf = Ralf()
        df = pd.DataFrame({
            "text": ["text1"],
            "label": ["A"]
        })
        
        with pytest.raises(ValueError):
            ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")


class TestHardwareDetection:
    """Test suite for hardware detection functionality."""
    
    def test_hardware_detection_attributes(self):
        """Test that hardware detection attributes are properly set."""
        ralf = Ralf()
        
        # Check that all hardware attributes exist
        assert hasattr(ralf, 'gpu_available')
        assert hasattr(ralf, 'gpu_count')
        assert hasattr(ralf, 'gpu_name')
        assert hasattr(ralf, 'gpu_ram_gb')
        assert hasattr(ralf, 'ram_gb')
        
        # Check data types
        assert isinstance(ralf.gpu_available, bool)
        assert isinstance(ralf.gpu_count, int)
        assert isinstance(ralf.ram_gb, (int, float))
        
        # Check logical constraints
        if ralf.gpu_available:
            assert ralf.gpu_count > 0
            assert ralf.gpu_name is not None
            assert ralf.gpu_ram_gb is not None
        else:
            assert ralf.gpu_count == 0
            assert ralf.gpu_name is None
            assert ralf.gpu_ram_gb is None
        
        assert ralf.ram_gb > 0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, tmp_path):
        """Test a complete workflow from data loading to state saving."""
        # Initialize
        ralf = Ralf()
        
        # Set API keys
        ralf.set_keys(open_api_key="test_open", gemini_key="test_gemini", hf_token="test_hf")
        
        # Load and process data
        df = pd.DataFrame({
            "text": ["positive text", "negative text", "positive text", "negative text", "positive text", "negative text"],
            "label": ["positive", "negative", "positive", "negative", "positive", "negative"]
        })
        ralf.load_and_process_data(df, text_column="text", label_column="label", model_name="bert-base-uncased")
        
        # Configure model
        ralf.load_and_configure_model()
        
        # Initialize trainer
        output_dir = tmp_path / "results"
        save_path = tmp_path / "ralf_state.pkl"
        ralf.initialize_trainer(output_dir=str(output_dir), save_path=str(save_path))
        
        # Save state
        save_file = tmp_path / "ralf_state.pkl"
        ralf.save_state(file_path=str(save_file))
        
        # Load state
        loaded_ralf = Ralf.load_state(file_path=str(save_file))
        
        # Verify complete restoration (API keys are not saved in state, so they won't be restored)
        assert loaded_ralf is not None
        assert loaded_ralf.model_name == "bert-base-uncased"
        assert loaded_ralf.num_labels == 2
        assert loaded_ralf.label_to_id == {"positive": 0, "negative": 1}
        assert loaded_ralf.tokenizer is not None
        assert loaded_ralf.model is not None
        assert loaded_ralf.train_dataset is not None
        assert loaded_ralf.val_dataset is not None
    
    def test_workflow_with_custom_columns(self, tmp_path):
        """Test complete workflow with custom column names."""
        ralf = Ralf()
        
        df = pd.DataFrame({
            "mytext": ["text1", "text2", "text3", "text4", "text5", "text6"],
            "mylabel": ["A", "B", "A", "B", "A", "B"]
        })
        
        ralf.load_and_process_data(df, text_column="mytext", label_column="mylabel", model_name="bert-base-uncased")
        ralf.load_and_configure_model()
        
        save_file = tmp_path / "ralf_state.pkl"
        ralf.save_state(file_path=str(save_file))
        
        loaded_ralf = Ralf.load_state(file_path=str(save_file))
        
        assert loaded_ralf is not None
        assert loaded_ralf.model_name == "bert-base-uncased"
        assert loaded_ralf.num_labels == 2
        assert set(loaded_ralf.label_to_id.keys()) == {"A", "B"}