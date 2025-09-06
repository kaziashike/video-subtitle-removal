# Video Subtitle Removal

An advanced solution for removing subtitles from videos using state-of-the-art AI techniques.

## Features
- Text Detection: PaddleOCR + DBNet for accurate subtitle detection
- Inpainting: LaMa for high-quality background reconstruction
- Temporal Consistency: RAFT Optical Flow to eliminate flickering
- Super Resolution: Real-ESRGAN to upscale video quality
- Edge Feathering: For seamless blending of inpainted regions

## Requirements
- Python 3.8-3.10
- CUDA-compatible GPU (recommended for performance)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kaziashike/video-subtitle-removal.git
cd video-subtitle-removal
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install External Libraries
The project requires three external libraries that need to be manually installed:

#### Option A: Clone repositories (Recommended)
```bash
# Create external_libs directory if it doesn't exist
mkdir external_libs
cd external_libs

# Clone the required repositories
git clone https://github.com/princeton-vl/RAFT.git
git clone https://github.com/advimman/lama.git
git clone https://github.com/xinntao/Real-ESRGAN.git

# Install dependencies for each library
cd RAFT
pip install -r requirements.txt
cd ../lama
pip install -r requirements.txt
cd ../Real-ESRGAN
pip install -r requirements.txt
cd ../..
```

#### Option B: Install as packages
```bash
pip install git+https://github.com/princeton-vl/RAFT.git
pip install git+https://github.com/advimman/lama.git
pip install git+https://github.com/xinntao/Real-ESRGAN.git
```

### 5. Download Pre-trained Models
You'll need to download pre-trained models for the external libraries:

#### RAFT Model
- Download [raft-things.pth](https://drive.google.com/uc?id=1sQP242j-5LV2CcbSU6v2jmC5IM8x4Ouy&export=download)
- Place it in `external_libs/RAFT/models/`

#### LaMa Model
- Download [big-lama](https://drive.google.com/uc?id=1SmYUd736F43rDLq58sDXKUBsBMQzIL5-&export=download)
- Extract and place the folder in `external_libs/lama/`

#### Real-ESRGAN Model
- Download [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- Place it in `external_libs/Real-ESRGAN/experiments/pretrained_models/`

### 6. GPU Support (Optional but Recommended)
For CUDA support, install the GPU versions of PyTorch and PaddlePaddle:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install paddlepaddle-gpu
```

## Usage
```bash
python main.py input_video.mp4 output_video.mp4
```

## Performance Notes
- Processing on CPU (like i5-6300U) can take several hours for a 1-minute video
- With a GPU, processing time can be reduced to minutes
- For best performance, use a modern GPU with CUDA support

## Project Structure
```
video-subtitle-removal/
├── main.py                 # Main processing pipeline
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── text_detection/         # Text detection module (PaddleOCR)
├── inpainting/             # Image inpainting module (LaMa)
├── temporal_consistency/   # Temporal consistency module (RAFT)
├── refinement/             # Image refinement module
├── super_resolution/       # Super resolution module (Real-ESRGAN)
├── external_libs/          # External libraries (RAFT, LaMa, Real-ESRGAN)
└── tests/                  # Test files
```

## Technical Advantages

1. **Better Accuracy**: PaddleOCR + DBNet outperforms EasyOCR in complex scenarios
2. **Professional Restoration**: LaMa provides superior inpainting compared to Stable Diffusion for real-world content
3. **Temporal Stability**: RAFT optical flow ensures consistent results across frames
4. **Enhanced Details**: Real-ESRGAN upscales to 1080p with sharp, natural details
5. **Modular Design**: Each component is independently maintainable and replaceable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.