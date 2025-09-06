# Video Subtitle Removal
python main.py "C:\Users\Itachi Ash\Downloads\test.mp4" output_video.mp4
A sophisticated solution for removing subtitles from videos with superior quality compared to vmake.ai. This project implements a complete pipeline that detects, removes, and restores video content with professional-grade results.

## Features

- **Advanced Text Detection**: Uses PaddleOCR + DBNet for precise subtitle identification, even in challenging conditions
- **Temporal Consistency**: Employs RAFT optical flow to prevent flickering between frames
- **Professional Inpainting**: Leverages LaMa (Large Mask Inpainting) for realistic content filling
- **Detail Enhancement**: Applies edge feathering and noise reduction for seamless results
- **Super Resolution**: Integrates Real-ESRGAN to upscale final output to 1080p with sharp details

## Prerequisites

- Python 3.8+
- pip
- git

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd video-subtitle-removal
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install basic requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Install external libraries:
   ```bash
   # Create directory for external libraries
   mkdir external_libs
   cd external_libs
   
   # Clone external repositories
   git clone https://github.com/advimman/lama.git
   git clone https://github.com/princeton-vl/RAFT.git
   git clone https://github.com/xinntao/Real-ESRGAN.git
   
   # Install each library
   cd lama && pip install -e . && cd ..
   cd RAFT && pip install -e . && cd ..
   cd Real-ESRGAN && pip install -e . && cd ..
   
   cd ..
   ```

5. Install additional dependencies:
   ```bash
   pip install torch torchvision
   ```

6. Download pre-trained models:
   - LaMa models (place in `lama/`)
   - RAFT models (place in `RAFT/models/`)
   - Real-ESRGAN models (place in `Real-ESRGAN/experiments/pretrained_models/`)

## Usage

```bash
python main.py input_video.mp4 output_video.mp4
```

## Pipeline Architecture

1. **Text Detection**
   - Uses PaddleOCR with DBNet for accurate subtitle detection
   - Handles rotated text, shadows, and low-contrast scenarios

2. **Temporal Consistency**
   - Implements RAFT optical flow for frame-to-frame consistency
   - Prevents flickering and temporal artifacts

3. **Mask Refinement**
   - Applies edge feathering for smooth blending
   - Uses morphological operations to clean up mask edges

4. **Inpainting**
   - Employs LaMa for high-quality content reconstruction
   - Superior to Stable Diffusion for real-world scenes

5. **Super Resolution**
   - Integrates Real-ESRGAN for detail enhancement
   - Upscales to 1080p with preserved sharpness

## Directory Structure

```
video-subtitle-removal/
├── main.py                 # Main application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── text_detection/         # Text detection module (PaddleOCR)
├── temporal_consistency/   # Temporal smoothing module (RAFT)
├── refinement/             # Mask refinement module
├── inpainting/             # Inpainting module (LaMa)
├── super_resolution/       # Super resolution module (Real-ESRGAN)
├── external_libs/          # External libraries (LaMa, RAFT, Real-ESRGAN)
└── README.md               # This file
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