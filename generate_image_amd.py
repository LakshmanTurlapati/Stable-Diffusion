#!/usr/bin/env python3
import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from datetime import datetime

# Check if Vulkan/AMD GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this device. This script is designed for AMD GPUs with Vulkan.")
print("AMD GPU detected! Using Vulkan acceleration.")

def convert_to_fp32(model):
    """Convert all model parameters to float32 for better compatibility."""
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    return model

def setup_sdxl_pipeline(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    """Set up the SDXL pipeline for AMD GPU."""
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

        # For AMD GPUs, we can use CUDA-compatible APIs
        pipe = pipe.to("cuda")
        
        # Use DPMSolverMultistepScheduler for faster inference
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable memory efficient attention if needed
        # pipe.enable_xformers_memory_efficient_attention()
        
        # For low VRAM GPUs, enable model offloading
        # pipe.enable_model_cpu_offload()
        
        print("Model loaded and ready.")
        return pipe
    
    except Exception as e:
        print(f"Error setting up pipeline: {e}")
        raise

def generate_image(pipe, prompt, negative_prompt="", height=512, width=512, steps=30, guidance_scale=7.5, seed=None):
    """Generate an image using the provided pipeline and parameters."""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 2147483647, (1,)).item())
        seed = generator.initial_seed()
    
    print(f"Generating image with seed: {seed}")
    print(f"Prompt: {prompt}")
    
    # Additional inference configuration
    inference_config = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    
    # Generate the image
    print("Generating image...")
    try:
        with torch.no_grad():
            image = pipe(**inference_config).images[0]
        print("Image generation complete!")
        return image, seed
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise

def save_image(image, output_dir="outputs"):
    """Save the generated image to a file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/generated_{timestamp}.png"
    
    # Save image
    image.save(filename)
    print(f"Image saved to: {filename}")
    return filename

def main():
    """Main function to run the image generation pipeline."""
    # Set up the pipeline
    pipe = setup_sdxl_pipeline()
    
    # Define the prompt
    prompt = input("Enter prompt for image generation: ")
    
    # Optional parameters
    negative_prompt = input("Enter negative prompt (optional, press Enter to skip): ")
    
    # Set default size
    height = 1024  # Default height for SDXL
    width = 1024   # Default width for SDXL
    
    # Generate the image
    image, seed = generate_image(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=30,
        guidance_scale=7.5,
    )
    
    # Save the image
    save_image(image)
    
    print(f"Image generation complete with seed {seed}.")
    
if __name__ == "__main__":
    main() 