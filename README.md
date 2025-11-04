# Plant Disease Detection using MobileNetV2 and Tkinter GUI

This project detects **plant diseases** (such as *Potato, Tomato,* and *Bell Pepper* diseases) using a **MobileNetV2** deep learning model trained on the **PlantVillage dataset**.  
It features a simple **Tkinter GUI** that allows users to upload leaf images and view instant predictions.

---

## Project Overview

The goal of this project is to help farmers and researchers quickly identify plant leaf diseases from images.  
By leveraging **transfer learning** with **MobileNetV2**, the model achieves high accuracy while remaining lightweight and fast enough for real-time use.

## Project Structure
plant-disease-detector/
│
├── models/
│ ├── mobilenetv2_best.h5 # trained CNN model
│ └── class_indices.npy # mapping between class indices and labels
│
├── gui/
│ ├── plant_disease_ui.py # main GUI application
│ └── background.jpg # GUI background image
│
├── requirements.txt # Python dependencies
└── README.md # documentation

##  Model Information

- **Architecture:** MobileNetV2 (transfer learning)
- **Framework:** TensorFlow / Keras
- **Trained on:** PlantVillage dataset (15 classes)
- **Accuracy:** ~85–90% (validation)
  
##  GUI Features
- Upload a leaf image (`.jpg`, `.png`, etc.)
- Automatically preprocesses and predicts disease
- Displays:
  - Leaf preview
  - Predicted class name
  - Confidence percentage
 Example:

##  How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. Run the GUI

python gui/plant_disease_ui.py

3.Upload a leaf image and view prediction on screen
## Example Predictions
| Image                            | Predicted Class            | Confidence |
| :------------------------------- | :------------------------- | :--------- |
| `tomato_early_blight.jpg`        | Tomato Early Blight        | 98.4%      |
| `potato_healthy.jpg`             | Potato Healthy             | 99.1%      |
| `bell_pepper_bacterial_spot.jpg` | Bell Pepper Bacterial Spot | 96.7%      |

## Requirements
tensorflow
keras
numpy
pillow
opencv-python
tkinter
## Notes

The dataset used for training (PlantVillage) is not included due to size.

The model (mobilenetv2_best.h5) and label file (class_indices.npy) are already provided in /models.

Works best with clear leaf images (centered and well-lit).
## Future Improvements

Deploy as a web or mobile app

Add more plant species

Integrate with real-time camera detection

Optimize for edge devices (Raspberry Pi, Jetson Nano
## Download Required Files
Before running the project, download the trained model and dataset from Google Drive using the following commands:
```bash
# Download model file
!gdown --id 1rGaB0ss-58sJrn9XDxmJF94FsXf42MO2 -O models/mobilenetv2_best.h5
# Download dataset 
!gdown --id 17U2xGvHzoXXh6OSJqLOWDW2RZePLe9h0 -O PlantVillage_split.zip
!unzip PlantVillage_split.zip -d ./PlantVillage_split
# Download class indices
!gdown --id 1Vql0DsCMsHCD2aO-6NRsDKnZN3Q12x-v -O models/class_indices.npy













