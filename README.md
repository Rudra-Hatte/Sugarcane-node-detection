
# 🌿 Sugarcane Node Detection using YOLOv8

This project aims to detect **nodes on sugarcane stalks** using a custom-trained YOLOv8 model. The system is capable of detecting nodes in both images and videos, and it can be extended to real-time detection for agricultural automation.

---

## 📁 Files Included

```
📦 Sugarcane-Node-Detection
 ├── train.py         # Training script for YOLOv8 using custom sugarcane dataset
 ├── detect.py        # Detection script for images, videos, or live feed
 ├── best.pt          # Trained model weights (YOLOv8 format)
 └── test.mp4         # Output video showing detection results
```

---

## 🧠 Project Overview

- **Model Used:** YOLOv8 (You Only Look Once - Ultralytics)
- **Objective:** Detect nodes on sugarcane using object detection
- **Frameworks:** Python, OpenCV, Ultralytics YOLOv8
- **Training Data:** Custom-annotated dataset (in YOLO format)
- **Output:** Bounding boxes over sugarcane nodes in images or videos

---

## 🔧 Setup Instructions

### 1. ✅ Install Requirements

Install Python dependencies using pip:

```bash
pip install ultralytics opencv-python
```

Alternatively, clone and set up Ultralytics from their official repo:

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

---

### 2. 🏋️‍♂️ Training the Model

To train the model on your custom sugarcane dataset:

```bash
python train.py
```

Make sure the dataset follows the YOLO format:

```
/dataset
 ├── images/
 │    ├── train/
 │    └── val/
 └── labels/
      ├── train/
      └── val/
```

Update dataset path and configuration inside `train.py`.

---

### 3. 🔍 Running Detection

Use the `detect.py` script for inference on images, videos, or webcam:

#### a. On an image:

```bash
python detect.py --source path/to/image.jpg --weights best.pt
```

#### b. On a video:

```bash
python detect.py --source path/to/video.mp4 --weights best.pt
```

#### c. On live webcam:

```bash
python detect.py --source 0 --weights best.pt
```

Detected nodes will be shown with bounding boxes and saved to the `runs/detect` folder by default.

---

## 🎥 Output Demo

The file `test.mp4` showcases a sample output of the trained model detecting sugarcane nodes in real-time video.

---

## 📊 Model Performance

- **Accuracy:** High performance on custom annotated dataset
- **Robustness:** Tested on various lighting and background conditions
- **Use-case:** Can be integrated into smart farming systems, yield estimation, and automated crop inspection.

---

## 👨‍🔬 Author

Developed by **Rudra Hatte**  
Third Year – Artificial Intelligence & Data Science  
Savitribai Phule Pune University (SPPU)

---

## 📬 Contact

Feel free to reach out for any queries or contributions:

- 📧 Email: rudrahatte.official@gmail.com  
- 🌐 GitHub: [github.com/rudrahatte](https://github.com/rudrahatte)

---

## ⭐ Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV for video/image handling
- Custom dataset manually annotated for sugarcane node detection

---

## ❤️ Support

If this project helped you, consider giving it a ⭐ on GitHub!
