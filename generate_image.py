#!/usr/bin/env python3
import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from datetime import datetime

# Check if MPS is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available on this device. This script is designed for Apple Silicon with MPS.")
print("Apple Silicon detected! Using MPS acceleration.")

def convert_to_fp32(model):
    """Convert all model parameters to float32 for MPS compatibility."""
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    return model

def setup_sdxl_pipeline(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    """Set up the SDXL pipeline for MPS."""
    print(f"Loading SDXL model: {model_id}")
    print("This may take a few minutes on first run as the model is downloaded...")

    # Configuration for loading the model
    load_config = {
        "torch_dtype": torch.float16,  # Load in fp16 as provided by the repository
        "use_safetensors": True,       # Use safetensors for faster loading
        "safety_checker": None,       # Disable safety checker (optional)
        "requires_safety_checker": False,
    }

    try:
        # Load the SDXL pipeline in fp16
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_config)

        # Convert all components to fp32 for MPS
        pipe.unet = convert_to_fp32(pipe.unet)
        pipe.vae = convert_to_fp32(pipe.vae)
        pipe.text_encoder = convert_to_fp32(pipe.text_encoder)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2 = convert_to_fp32(pipe.text_encoder_2)

        # Set up the scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Move the pipeline to MPS
        pipe.to("mps")
        print("Pipeline moved to MPS device.")

        # Perform a warm-up pass to initialize MPS (helps avoid initial runtime errors)
        print("Running warm-up pass on MPS...")
        warmup_params = {
            "prompt": "warm up",
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "output_type": "np",
            "added_cond_kwargs": {
                "text_embeds": torch.zeros(1, 1280, device="mps", dtype=torch.float32),
                "time_ids": torch.tensor([[512, 512, 512, 512]], device="mps", dtype=torch.float32),
            }
        }
        with torch.inference_mode():
            try:
                _ = pipe(**warmup_params)
                print("Warm-up pass completed.")
            except Exception as e:
                print(f"Warm-up pass failed (ignoring): {str(e)[:100]}...")

        # Clear MPS cache if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        return pipe

    except Exception as e:
        raise RuntimeError(f"Failed to set up SDXL pipeline: {str(e)}")

def generate_image(pipe, prompt, negative_prompt="", height=512, width=512, steps=30, guidance_scale=7.5, seed=None):
    """Generate an image using the SDXL pipeline on MPS."""
    device = "mps"
    
    # Set up generation parameters
    generation_params = {
        "prompt": prompt + ", photorealistic, ultra detailed, sharp focus, cinematic lighting",
        "negative_prompt": negative_prompt if negative_prompt else "cartoon, anime, 3d, painting, drawing, blurry, deformed, disfigured",
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "output_type": "pil",  # Return PIL image
        "added_cond_kwargs": {
            "text_embeds": torch.zeros(1, 1280, device=device, dtype=torch.float32),
            "time_ids": torch.tensor([[height, width, height, width]], device=device, dtype=torch.float32),
        }
    }

    # Handle random seed
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)  # MPS uses CPU-based generator
        generation_params["generator"] = generator
        print(f"Using seed: {seed}")
    else:
        print("Using random seed")

    try:
        # Clear MPS cache before generation
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        print("Generating image with MPS acceleration...")
        with torch.inference_mode():
            result = pipe(**generation_params)
            return result.images[0]

    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def save_image(image, output_dir="outputs"):
    """Save the generated image with a timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sdxl_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"Image saved to {filepath}")
    return filepath

def main():
    """Main function to run the SDXL image generation on MPS."""
    # Set up the pipeline once
    pipe = setup_sdxl_pipeline()
    
    while True:
        try:
            # Get user input with exit option
            prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            if prompt.lower() in ['quit', 'exit', '']:
                print("Exiting...")
                break

            # Keep existing parameter collection
            negative_prompt = input("Enter negative prompt (Enter for default): ").strip()
            height = int(input("Enter image height (default: 1024): ") or 1024)
            width = int(input("Enter image width (default: 1024): ") or 1024)
            steps = int(input("Number of steps (default: 20): ") or 20)
            guidance_scale = float(input("Guidance scale (default: 5.0): ") or 5.0)
            seed_input = input("Random seed (Enter for random): ").strip()
            seed = int(seed_input) if seed_input else None

            # Generate and save the image
            image = generate_image(pipe, prompt, negative_prompt, height, width, steps, guidance_scale, seed)
            save_image(image)
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue

        print("Ready for next prompt...")

    print("Image generation complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")