
# FastSAM Setup Guide

This guide provides the setup instructions to install and configure the FastSAM environment.

## Prerequisites

- **Python Version**: 3.9.11  
  Ensure you are using Python 3.9.11 as the specified version for compatibility.

## Setup Instructions

1. **Clone the FastSAM Repository**
   ```bash
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. **Install Dependencies**

   - Create a `requirements.txt` file in the main directory, and add all necessary library names and versions required by FastSAM.
   - Install all dependencies from `requirements.txt`:
   
     ```bash
     pip install -r FastSAM/requirements.txt
     ```

   - Install additional required packages:
     ```bash
     pip install git+https://github.com/openai/CLIP.git
     pip install git+https://github.com/facebookresearch/segment-anything.git
     pip install roboflow supervision jupyter_bbox_widget
     ```

3. **Download Weights**

   Create a directory to store the weights and download the required weights files:

   ```bash
   mkdir -p $HOME/weights
   wget -P $HOME/weights https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
   wget -P $HOME/weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

4. **Verify Weights Download**

   List the downloaded weights to confirm:

   ```bash
   ls -lh $HOME/weights
   ```

5. **Set Checkpoint Paths**

   Define the checkpoint paths in your script or environment:
   
   ```python
   FAST_SAM_CHECKPOINT_PATH = f"{HOME}/weights/FastSAM.pt"
   SAM_SAM_CHECKPOINT_PATH = f"{HOME}/weights/sam_vit_h_4b8939.pth"
   ```

--- 

This guide provides a straightforward setup for running FastSAM in a Python 3.9.11 environment. Make sure to follow each step carefully to ensure all dependencies and weights are properly installed and accessible.
