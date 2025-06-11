import pytest
import torch
from PIL import Image
import sys
import os

# Add src to path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import VeS

@pytest.fixture(scope="module")
def model():
    """Fixture to initialize the VeS model once per test module."""
    # Running on CPU for testing to avoid GPU memory issues in CI/CD
    # and to allow testing on machines without a GPU.
    model = VeS(use_amp=False).to("cpu")
    model.eval()  # Set to eval mode for testing
    return model

@pytest.fixture
def dummy_inputs():
    """Fixture to create dummy audio and image inputs."""
    batch_size = 2
    audio_seq_len = 16000 * 5  # 5 seconds of audio at 16kHz
    
    # Dummy audio: (B, T)
    audio_input = torch.randn(batch_size, audio_seq_len)
    
    # Dummy image: list of PIL Images
    #images = [Image.new('RGB', (224, 224)) for _ in range(batch_size)]
    images = torch.randn(batch_size, 3, 224, 224)
    
    return audio_input, images

def test_model_initialization(model):
    """Test if the VeS model initializes correctly."""
    assert model is not None
    assert isinstance(model, VeS)
    print("Model initialized successfully.")

def test_forward_pass(model, dummy_inputs):
    """Test the forward pass of the VeS model."""
    audio_input, images = dummy_inputs
    
    with torch.no_grad():
        outputs = model(audio_input, images)

    assert isinstance(outputs, dict)
    print("Forward pass completed successfully.")
    
    # --- Check output dictionary keys ---
    expected_keys = ['loss', 'clip_sims', 'audio_feats', 'visual_feats', 'audio_attention_mask']
    for key in expected_keys:
        assert key in outputs
    print(f"Output contains all expected keys: {expected_keys}")

    # --- Check output shapes ---
    batch_size = len(images)
    embedding_dim_vision = model.visual_embedder.projection2.out_features
    embedding_dim_audio = model.audio_embedder.projection2.out_features

    # loss
    assert outputs['loss'].shape == torch.Size([])
    print(f"Loss shape is correct: {outputs['loss'].shape}")

    # clip_sims
    assert outputs['clip_sims'].shape == (batch_size, batch_size)
    print(f"Clip sims shape is correct: {outputs['clip_sims'].shape}")

    # audio_feats
    # Shape is (B, Na, D) - Na is variable, so we check B and D
    assert outputs['audio_feats'].shape[0] == batch_size
    assert outputs['audio_feats'].shape[2] == embedding_dim_audio
    print(f"Audio feats shape is correct (checked B and D): {outputs['audio_feats'].shape}")

    # visual_feats
    # Shape is (B, Nv, D) - Nv is fixed for a given image size
    assert outputs['visual_feats'].shape[0] == batch_size
    assert outputs['visual_feats'].shape[2] == embedding_dim_vision
    print(f"Visual feats shape is correct (checked B, Nv, D): {outputs['visual_feats'].shape}")

    # audio_attention_mask
    # Shape is (B, Na) - Na should match audio_feats
    assert outputs['audio_attention_mask'].shape[0] == batch_size
    assert outputs['audio_attention_mask'].shape[1] == outputs['audio_feats'].shape[1]
    print(f"Audio attention mask shape is correct: {outputs['audio_attention_mask'].shape}")

def test_batch_size_one(model):
    """Test forward pass with a batch size of 1."""
    audio_input = torch.randn(1, 16000 * 2) # 2 seconds
    images = [Image.new('RGB', (224, 224))]

    with torch.no_grad():
        outputs = model(audio_input, images)

    assert outputs['loss'] is not None
    assert outputs['clip_sims'].shape == (1, 1)
    print("Forward pass with batch size 1 is successful.")

if __name__ == "__main__":
    pytest.main([__file__]) 