import unittest
import pandas as pd
import tempfile
import os
import pickle
from unittest.mock import patch, MagicMock
import torch

from ralf import Ralf, TrainerConfig


class TestRalf(unittest.TestCase):
    """Test cases for the Ralf class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ralf = Ralf()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'text': [
                'This is a positive review',
                'This is a negative review',
                'This is a neutral review',
                'Another positive example',
                'Another negative example'
            ],
            'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })

    def test_initialization(self):
        """Test Ralf class initialization."""
        self.assertIsNotNone(self.ralf)
        self.assertIsNone(self.ralf.golden_dataset)
        self.assertIsNone(self.ralf.platinum_dataset)
        self.assertIsNone(self.ralf.model_name)
        self.assertIsNone(self.ralf.trainer)
        self.assertIsNone(self.ralf.num_labels)
        self.assertIsNone(self.ralf.label_to_id)
        self.assertIsNone(self.ralf.id_to_label)
        self.assertIsNone(self.ralf.tokenizer)
        self.assertIsNone(self.ralf.train_dataset)
        self.assertIsNone(self.ralf.val_dataset)
        self.assertIsNone(self.ralf.model)
        
        # Test API keys initialization
        self.assertIsNone(self.ralf.open_api_key)
        self.assertIsNone(self.ralf.gemini_key)
        self.assertIsNone(self.ralf.hf_token)
        
        # Test hardware detection
        self.assertIsInstance(self.ralf.gpu_available, bool)
        self.assertIsInstance(self.ralf.gpu_count, int)
        self.assertIsInstance(self.ralf.ram_gb, float)

    def test_set_keys(self):
        """Test setting API keys."""
        open_api_key = "test_openai_key"
        gemini_key = "test_gemini_key"
        hf_token = "test_hf_token"
        
        self.ralf.set_keys(
            open_api_key=open_api_key,
            gemini_key=gemini_key,
            hf_token=hf_token
        )
        
        self.assertEqual(self.ralf.open_api_key, open_api_key)
        self.assertEqual(self.ralf.gemini_key, gemini_key)
        self.assertEqual(self.ralf.hf_token, hf_token)

    @patch('ralf.ralf.AutoTokenizer.from_pretrained')
    @patch('ralf.ralf.train_test_split')
    @patch('ralf.ralf.Dataset.from_pandas')
    def test_load_and_process_data(self, mock_dataset_from_pandas, mock_split, mock_tokenizer):
        """Test data loading and processing."""
        # Mock train_test_split to return sample data
        train_df = self.sample_data.iloc[:3]
        val_df = self.sample_data.iloc[3:]
        mock_split.return_value = (train_df, val_df)
        
        # Mock dataset creation - need 3 calls: original dataset, train dataset, val dataset
        mock_original_dataset = MagicMock()
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_dataset_from_pandas.side_effect = [mock_original_dataset, mock_train_dataset, mock_val_dataset]
        
        # Mock tokenizer to return proper tokenization results
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            'input_ids': [[1, 2, 3, 4, 5]] * len(texts),
            'attention_mask': [[1, 1, 1, 1, 1]] * len(texts)
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock the map method to return the same dataset
        mock_train_dataset.map.return_value = mock_train_dataset
        mock_val_dataset.map.return_value = mock_val_dataset
        
        # Mock remove_columns to return the same dataset
        mock_train_dataset.remove_columns.return_value = mock_train_dataset
        mock_val_dataset.remove_columns.return_value = mock_val_dataset
        
        # Test data loading
        self.ralf.load_and_process_data(
            df=self.sample_data,
            text_column='text',
            label_column='label',
            model_name='bert-base-uncased'
        )
        
        # Verify model_name is set
        self.assertEqual(self.ralf.model_name, 'bert-base-uncased')
        
        # Verify label mappings
        self.assertEqual(self.ralf.num_labels, 3)
        expected_label_to_id = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.assertEqual(self.ralf.label_to_id, expected_label_to_id)
        
        # Verify datasets are created
        self.assertIsNotNone(self.ralf.train_dataset)
        self.assertIsNotNone(self.ralf.val_dataset)
        
        # Verify tokenizer is initialized
        mock_tokenizer.assert_called_once_with('bert-base-uncased')

    def test_load_and_process_data_without_model_name(self):
        """Test that load_and_process_data raises error without model_name."""
        with self.assertRaises(ValueError):
            self.ralf.load_and_process_data(
                df=self.sample_data,
                text_column='text',
                label_column='label',
                model_name=None
            )

    @patch('ralf.ralf.AutoModelForSequenceClassification.from_pretrained')
    @patch('ralf.ralf.get_peft_model')
    def test_load_and_configure_model(self, mock_get_peft_model, mock_model_from_pretrained):
        """Test model loading and LoRA configuration."""
        # Set up prerequisites
        self.ralf.model_name = 'bert-base-uncased'
        self.ralf.num_labels = 3
        
        # Mock model
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_get_peft_model.return_value = mock_model
        
        # Test model loading
        self.ralf.load_and_configure_model()
        
        # Verify model is loaded
        mock_model_from_pretrained.assert_called_once_with(
            'bert-base-uncased', 
            num_labels=3
        )
        
        # Verify LoRA is applied
        mock_get_peft_model.assert_called_once()
        
        # Verify model is set
        self.assertEqual(self.ralf.model, mock_model)

    def test_load_and_configure_model_without_model_name(self):
        """Test that load_and_configure_model fails without model_name."""
        self.ralf.num_labels = 3
        # The actual error is OSError when model_name is None, not AttributeError
        with self.assertRaises(OSError):
            self.ralf.load_and_configure_model()

    @patch('ralf.ralf.TrainingArguments')
    @patch('ralf.ralf.Trainer')
    @patch('ralf.ralf.RalfSavingCallback')
    def test_initialize_trainer(self, mock_callback, mock_trainer, mock_training_args):
        """Test trainer initialization."""
        # Set up prerequisites
        self.ralf.model = MagicMock()
        self.ralf.train_dataset = MagicMock()
        self.ralf.val_dataset = MagicMock()
        self.ralf.tokenizer = MagicMock()
        
        # Mock TrainingArguments
        mock_args = MagicMock()
        mock_training_args.return_value = mock_args
        
        # Mock Trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock callback
        mock_callback_instance = MagicMock()
        mock_callback.return_value = mock_callback_instance
        
        # Create config
        config = TrainerConfig(
            output_dir="./test_results",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8
        )
        
        # Test trainer initialization
        self.ralf.initialize_trainer(config)
        
        # Verify TrainingArguments is called with config dict
        mock_training_args.assert_called_once()
        call_args = mock_training_args.call_args[1]  # Get kwargs
        self.assertEqual(call_args['output_dir'], "./test_results")
        self.assertEqual(call_args['num_train_epochs'], 1)
        self.assertEqual(call_args['per_device_train_batch_size'], 8)
        self.assertEqual(call_args['per_device_eval_batch_size'], 8)
        
        # Verify Trainer is called
        mock_trainer.assert_called_once()
        
        # Verify trainer is set
        self.assertEqual(self.ralf.trainer, mock_trainer_instance)

    def test_save_state(self):
        """Test state saving functionality."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Set some test data
            self.ralf.model_name = 'test-model'
            self.ralf.num_labels = 3
            
            # Test saving
            self.ralf.save_state(temp_path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))
            
            # Verify file can be loaded
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.assertEqual(loaded_data.model_name, 'test-model')
                self.assertEqual(loaded_data.num_labels, 3)
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_state_success(self):
        """Test successful state loading."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save test state
            self.ralf.model_name = 'test-model'
            self.ralf.num_labels = 3
            self.ralf.save_state(temp_path)
            
            # Test loading
            loaded_ralf = Ralf.load_state(temp_path)
            
            # Verify loaded data
            self.assertIsNotNone(loaded_ralf)
            self.assertEqual(loaded_ralf.model_name, 'test-model')
            self.assertEqual(loaded_ralf.num_labels, 3)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_state_file_not_found(self):
        """Test state loading when file doesn't exist."""
        result = Ralf.load_state("nonexistent_file.pkl")
        self.assertIsNone(result)

    @patch('ralf.ralf.AutoModelForSequenceClassification.from_pretrained')
    @patch('ralf.ralf.get_peft_model')
    @patch('ralf.ralf.AutoTokenizer.from_pretrained')
    @patch('ralf.ralf.TrainingArguments')
    @patch('ralf.ralf.Trainer')
    @patch('ralf.ralf.RalfSavingCallback')
    def test_restore_non_picklable(self, mock_callback, mock_trainer, mock_training_args, 
                                  mock_tokenizer, mock_get_peft_model, mock_model_from_pretrained):
        """Test restoring non-picklable attributes."""
        # Set up prerequisites
        self.ralf.model_name = 'bert-base-uncased'
        self.ralf.num_labels = 3
        self.ralf.train_dataset = MagicMock()
        self.ralf.val_dataset = MagicMock()
        
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_get_peft_model.return_value = mock_model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock trainer components
        mock_args = MagicMock()
        mock_training_args.return_value = mock_args
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_callback_instance = MagicMock()
        mock_callback.return_value = mock_callback_instance
        
        # Create config
        config = TrainerConfig()
        
        # Test restoration
        self.ralf.restore_non_picklable(config)
        
        # Verify model is loaded
        mock_model_from_pretrained.assert_called_once_with(
            'bert-base-uncased', 
            num_labels=3
        )
        
        # Verify tokenizer is loaded
        mock_tokenizer.assert_called_once_with('bert-base-uncased')
        
        # Verify trainer is initialized
        mock_trainer.assert_called_once()

    def test_getstate_setstate(self):
        """Test pickling and unpickling behavior."""
        # Set some test data
        self.ralf.model_name = 'test-model'
        self.ralf.num_labels = 3
        self.ralf.model = MagicMock()  # Non-picklable
        self.ralf.trainer = MagicMock()  # Non-picklable
        self.ralf.tokenizer = MagicMock()  # Non-picklable
        
        # Test __getstate__
        state = self.ralf.__getstate__()
        
        # Verify non-picklable attributes are None
        self.assertIsNone(state['model'])
        self.assertIsNone(state['trainer'])
        self.assertIsNone(state['tokenizer'])
        
        # Verify other attributes are preserved
        self.assertEqual(state['model_name'], 'test-model')
        self.assertEqual(state['num_labels'], 3)
        
        # Test __setstate__
        new_ralf = Ralf()
        new_ralf.__setstate__(state)
        
        # Verify state is restored
        self.assertEqual(new_ralf.model_name, 'test-model')
        self.assertEqual(new_ralf.num_labels, 3)
        self.assertIsNone(new_ralf.model)
        self.assertIsNone(new_ralf.trainer)
        self.assertIsNone(new_ralf.tokenizer)


class TestTrainerConfig(unittest.TestCase):
    """Test cases for the TrainerConfig class."""

    def test_default_values(self):
        """Test TrainerConfig default values."""
        config = TrainerConfig()
        
        self.assertEqual(config.output_dir, "./results")
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.per_device_train_batch_size, 16)
        self.assertEqual(config.per_device_eval_batch_size, 16)
        self.assertEqual(config.warmup_steps, 500)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.logging_dir, "./logs")
        self.assertEqual(config.logging_steps, 10)
        self.assertEqual(config.eval_strategy, "epoch")
        self.assertEqual(config.save_strategy, "epoch")
        self.assertTrue(config.load_best_model_at_end)
        self.assertEqual(config.metric_for_best_model, "eval_loss")
        self.assertFalse(config.greater_is_better)
        self.assertEqual(config.report_to, "none")
        self.assertEqual(config.save_path, "ralf_state.pkl")

    def test_custom_values(self):
        """Test TrainerConfig with custom values."""
        config = TrainerConfig(
            output_dir="./custom_results",
            num_train_epochs=5,
            per_device_train_batch_size=32,
            save_path="custom_state.pkl"
        )
        
        self.assertEqual(config.output_dir, "./custom_results")
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.per_device_train_batch_size, 32)
        self.assertEqual(config.save_path, "custom_state.pkl")
        
        # Verify other defaults are preserved
        self.assertEqual(config.per_device_eval_batch_size, 16)
        self.assertEqual(config.warmup_steps, 500)

    def test_dict_conversion(self):
        """Test that TrainerConfig can be converted to dict for TrainingArguments."""
        config = TrainerConfig(
            output_dir="./test_results",
            num_train_epochs=1,
            per_device_train_batch_size=8
        )
        
        # Use model_dump() instead of dict() for Pydantic v2
        config_dict = config.model_dump()
        
        self.assertEqual(config_dict['output_dir'], "./test_results")
        self.assertEqual(config_dict['num_train_epochs'], 1)
        self.assertEqual(config_dict['per_device_train_batch_size'], 8)
        self.assertIn('save_path', config_dict)  # Should be included


if __name__ == '__main__':
    unittest.main()
