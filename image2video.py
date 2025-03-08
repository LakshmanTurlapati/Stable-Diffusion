#!/usr/bin/env python3
import os
import torch
from diffusers import StableVideoDiffusionPipeline
from datetime import datetime
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Check if MPS is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available on this device. This script is designed for Apple Silicon with MPS.")
print("Apple Silicon detected! Using MPS acceleration.")

def convert_to_fp32(model):
    """Convert all model parameters to float32 for MPS compatibility."""
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    return model

def setup_video_diffusion_pipeline(model_id="stabilityai/stable-video-diffusion-img2vid-xt"):
    """Set up the Stable Video Diffusion pipeline for MPS."""
    print(f"Loading Stable Video Diffusion model: {model_id}")
    print("This may take a few minutes on first run as the model is downloaded...")

    # Configuration for loading the model
    load_config = {
        "torch_dtype": torch.float16,  # Load in fp16 as provided by the repository
        "variant": "fp16",
        "use_safetensors": True,       # Use safetensors for faster loading
    }

    try:
        # Load the Stable Video Diffusion pipeline in fp16
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, **load_config)

        # Convert components to fp32 for MPS compatibility
        pipe.unet = convert_to_fp32(pipe.unet)
        pipe.vae = convert_to_fp32(pipe.vae)
        pipe.image_encoder = convert_to_fp32(pipe.image_encoder)

        # Move the pipeline to MPS
        pipe.to("mps")
        print("Pipeline moved to MPS device.")

        # Clear MPS cache if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        return pipe

    except Exception as e:
        raise RuntimeError(f"Failed to set up Stable Video Diffusion pipeline: {str(e)}")

def generate_video(pipe, image_path, motion_bucket_id=127, fps=7, num_frames=25, noise_aug_strength=0.02, seed=None):
    """Generate a video from an image using the Stable Video Diffusion pipeline on MPS."""
    device = "mps"
    
    # Load the input image
    image = Image.open(image_path).convert("RGB")
    
    # Set up generation parameters
    generation_params = {
        "image": image,
        "motion_bucket_id": motion_bucket_id,
        "fps": fps,
        "noise_aug_strength": noise_aug_strength,
        "num_frames": num_frames,
        "output_type": "pil",  # Return PIL image sequence
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

        print("Generating video with MPS acceleration...")
        with torch.inference_mode():
            frames = pipe(**generation_params).frames[0]
            return frames

    except Exception as e:
        raise RuntimeError(f"Video generation failed: {str(e)}")

def save_video(frames, output_dir="outputs", fps=7):
    """Save the generated frames as a video and as individual images."""
    import imageio
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as MP4
    video_filename = f"svd_{timestamp}.mp4"
    video_filepath = os.path.join(output_dir, video_filename)
    
    # Convert frames to numpy arrays and save as video
    frame_arrays = [frame if isinstance(frame, Image.Image) else Image.fromarray(frame) for frame in frames]
    imageio.mimsave(video_filepath, frame_arrays, fps=fps)
    
    # Create a frames directory
    frames_dir = os.path.join(output_dir, f"svd_{timestamp}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Save individual frames
    for i, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            frame_image = frame
        else:
            frame_image = Image.fromarray(frame)
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame_image.save(frame_path)
    
    print(f"Video saved to {video_filepath}")
    print(f"Frames saved to {frames_dir}")
    return video_filepath

def select_image_file():
    """Open a file dialog to select an image file."""
    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the main window

    # Set up file dialog options
    file_types = [
        ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
        ('All files', '*.*')
    ]

    # Show the file dialog
    file_path = askopenfilename(
        title="Select an input image",
        filetypes=file_types
    )
    
    # Destroy the Tkinter root window
    root.destroy()
    
    return file_path

def main():
    """Main function to run the Stable Video Diffusion generation on MPS."""
    # Set up the pipeline once
    pipe = setup_video_diffusion_pipeline()
    
    while True:
        try:
            # Get user input with exit option
            print("\nSelect an input image (or type 'quit' to exit)")
            image_path = select_image_file()
            
            if not image_path:
                print("No file selected. Please try again.")
                continue
                
            if image_path.lower() in ['quit', 'exit', '']:
                print("Exiting...")
                break
                
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            # Get video generation parameters
            motion_bucket_id = int(input("Enter motion bucket ID (0-255, higher = more motion, default: 127): ") or 127)
            fps = int(input("Enter FPS (default: 7): ") or 7)
            num_frames = int(input("Number of frames (default: 25): ") or 25)
            noise_aug_strength = float(input("Noise augmentation strength (0.0-1.0, default: 0.02): ") or 0.02)
            seed_input = input("Random seed (Enter for random): ").strip()
            seed = int(seed_input) if seed_input else None

            # Generate and save the video
            frames = generate_video(pipe, image_path, motion_bucket_id, fps, num_frames, noise_aug_strength, seed)
            save_video(frames, fps=fps)
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue

        print("Ready for next input...")

    print("Video generation complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
