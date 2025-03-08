# Stable Diffusion XL Image Generator

A script for generating images using Stable Diffusion XL optimized for Apple Silicon with MPS acceleration.

## Requirements

- macOS with Apple Silicon (M1, M2, M3 chip)
- Python 3.8+
- At least 16GB of RAM recommended
- About 10GB of disk space for the model and environment

## Setup

1. Make the setup script executable:
   ```
   chmod +x setup.sh
   ```

2. Run the setup script:
   ```
   ./setup.sh
   ```

   This will:
   - Create a virtual environment
   - Install all required dependencies

3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

## Usage

Run the image generation script:
```
python generate_image.py
```

Follow the interactive prompts to:
- Enter a prompt describing the image you want to generate
- Optionally provide a negative prompt
- Adjust generation parameters (height, width, steps, guidance scale, seed)

Generated images will be saved in the `outputs` directory.

## Notes

- The first run will download the SDXL model (approximately 6GB)
- Generation can take several minutes depending on your machine and selected parameters
- Higher resolution images and more inference steps will increase generation time 