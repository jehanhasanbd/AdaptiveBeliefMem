# scripts/test_setup.py
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import json


def test_setup():
    """Test basic setup without any training."""

    print("=" * 50)
    print("TESTING SETUP")
    print("=" * 50)

    # Test 1: GPU availability
    print("\n1. GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test 2: Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    print("   ✓ Tokenizer loaded")

    # Test 3: Load model in 4-bit
    print("\n3. Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModel.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print(f"   ✓ Model loaded on: {model.device}")

    # Test 4: Load data
    print("\n4. Loading test data...")
    with open("data/processed/test.json", 'r') as f:
        test_data = json.load(f)
    print(f"   ✓ Loaded {len(test_data)} examples")

    # Test 5: Simple encoding
    print("\n5. Testing encoding...")
    test_text = ["This is a test sentence."]
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    print(f"   ✓ Encoding successful, embedding shape: {embedding.shape}")

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYour setup is working correctly.")
    print("The device issue was fixed by moving all tensors to GPU.")


if __name__ == "__main__":
    test_setup()