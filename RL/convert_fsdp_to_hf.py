#!/usr/bin/env python3
"""
Convert FSDP checkpoint to HuggingFace format
"""

import sys
checkpoint_dir = sys.argv[1]
output_dir = sys.argv[2]
hf_config_dir = sys.argv[3]

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig
import os

def main():
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Check if fsdp_config.json exists
    fsdp_config = os.path.join(checkpoint_dir, "fsdp_config.json")
    if not os.path.exists(fsdp_config):
        print(f"Error: fsdp_config.json not found in {checkpoint_dir}")
        sys.exit(1)

    # Check if huggingface config exists
    if not os.path.exists(hf_config_dir):
        print(f"Error: HuggingFace config directory not found: {hf_config_dir}")
        sys.exit(1)

    print(f"Checkpoint found: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create merger config
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_dir,
        target_dir=output_dir,
        trust_remote_code=True,
        hf_model_config_path=hf_config_dir,
        use_cpu_initialization=False
    )

    print("\n" + "="*60)
    print("Starting FSDP to HuggingFace conversion...")
    print("="*60 + "\n")

    try:
        # Initialize merger
        merger = FSDPModelMerger(config)

        # Execute merge
        merger.merge_and_save()

        # Cleanup
        merger.cleanup()

        print("\n" + "="*60)
        print(f"Conversion completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("="*60)

        # List output files
        print("\nOutput files:")
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
            else:
                print(f"  - {file}/ (directory)")

    except Exception as e:
        print(f"\nConversion failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
