
# Setup Guide

Setup instructions to install and configure the environment.

## Prerequisites

- **Python Version**: 3.9.11  
  Ensure you are using Python 3.9.11 as the specified version for compatibility.

## Setup Instructions

1. **Install Dependencies**

   - Install all dependencies from `requirements.txt`:
   
     ```bash
     pip install -r requirements.txt
     ```

   - Install additional required packages:
     ```bash
     pip install git+https://github.com/openai/CLIP.git
     pip install git+https://github.com/facebookresearch/segment-anything.git
     pip install roboflow supervision jupyter_bbox_widget
     ```

2. **Download Weights**

   Create a directory to store the weights and download the required weights files:

   ```bash
   mkdir -p /weights
   wget -P /weights https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
   wget -P /weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```
