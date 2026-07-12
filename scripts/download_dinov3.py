import argparse
import os
from transformers import AutoModel
import sys

def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 model")
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    args = parser.parse_args()
    
    print(f"Downloading model {args.model_name}")
    try:
        model = AutoModel.from_pretrained(args.model_name)
        print("Success! Weights cached in Hugging Face cache.")
    except Exception as e:
        print(f"Failed to download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
