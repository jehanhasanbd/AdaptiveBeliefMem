# scripts/convert_checkpoint.py
import torch
from pathlib import Path


def convert_checkpoint():
    """Convert old checkpoint format to new format."""

    # Path to your existing checkpoint
    old_checkpoint_path = Path("outputs/checkpoints/checkpoint_epoch_0.pt")

    if not old_checkpoint_path.exists():
        print(f"Checkpoint not found: {old_checkpoint_path}")
        return

    print(f"Loading checkpoint from: {old_checkpoint_path}")
    checkpoint = torch.load(old_checkpoint_path, map_location="cpu")

    print(f"Original checkpoint keys: {checkpoint.keys()}")

    # Create new checkpoint with proper format
    new_checkpoint = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_accuracy": checkpoint.get("best_accuracy", 0.0),
    }

    # Handle builder state - extract LoRA weights if present
    if "builder_state" in checkpoint:
        # Old format - try to extract just the LoRA parts
        builder_state = checkpoint["builder_state"]

        # Filter for LoRA parameters (they contain "lora")
        lora_state = {k: v for k, v in builder_state.items() if "lora" in k.lower()}

        if lora_state:
            new_checkpoint["builder_lora_state"] = lora_state
            print(f"Extracted {len(lora_state)} LoRA parameters")
        else:
            # If no LoRA params, create empty dict
            new_checkpoint["builder_lora_state"] = {}
            print("No LoRA parameters found, using empty dict")
    else:
        new_checkpoint["builder_lora_state"] = {}

    # Handle uncertainty and retriever states
    new_checkpoint["uncertainty_state"] = checkpoint.get("uncertainty_state", {})
    new_checkpoint["retriever_state"] = checkpoint.get("retriever_state", {})
    new_checkpoint["optimizer_state"] = checkpoint.get("optimizer_state", {})

    # Save as best_model
    new_checkpoint_path = Path("outputs/checkpoints/best_model_3b.pt")
    torch.save(new_checkpoint, new_checkpoint_path)

    print(f"\nConverted checkpoint saved to: {new_checkpoint_path}")
    print(f"New checkpoint keys: {new_checkpoint.keys()}")

    # Also save a backup of original with new name
    backup_path = Path("outputs/checkpoints/checkpoint_epoch_0_backup.pt")
    import shutil
    shutil.copy(old_checkpoint_path, backup_path)
    print(f"Original backed up to: {backup_path}")

    return new_checkpoint_path


def verify_checkpoint(checkpoint_path):
    """Verify the checkpoint can be loaded properly."""
    print(f"\nVerifying: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_keys = ["builder_lora_state", "uncertainty_state", "retriever_state"]
    missing = [k for k in required_keys if k not in checkpoint]

    if missing:
        print(f"❌ Missing keys: {missing}")
        return False
    else:
        print(f"✅ All required keys present")
        print(f"   - builder_lora_state: {len(checkpoint['builder_lora_state'])} parameters")
        print(f"   - uncertainty_state: {len(checkpoint['uncertainty_state'])} parameters")
        print(f"   - retriever_state: {len(checkpoint['retriever_state'])} parameters")
        return True


if __name__ == "__main__":
    # Convert checkpoint
    converted_path = convert_checkpoint()

    # Verify the converted checkpoint
    if converted_path:
        verify_checkpoint(converted_path)

        print("\n" + "=" * 50)
        print("✅ CONVERSION COMPLETE")
        print("=" * 50)
        print("\nNow you can run:")
        print(f"python scripts/run_experiment.py --mode eval --checkpoint {converted_path}")