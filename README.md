<h1 align=center><font size = 6>Drone Detection and Tracking using YOLOv11x</font></h1>

<img src="https://images.pexels.com/photos/724921/pexels-photo-724921.jpeg" height=550 width=1000 alt="https://www.pexels.com/"/>

<small>Picture Source: <a href="https://www.pexels.com/@pok-rie-33563/">Pok Rie</a></small>

<br>

## **Abstract**

Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used in sectors ranging from surveillance and delivery to agriculture. While offering significant benefits, their proliferation also raises concerns about security, privacy, and airspace safety. This project presents a robust, real-time drone detection and tracking system based on **YOLOv11x**, the latest iteration of the YOLO (You Only Look Once) object detection family. Leveraging architectural innovations like **C3k2 blocks**, **SPPF**, and **C2PSA spatial attention**, the model excels at detecting small, fast-moving aerial targets in diverse and complex environments.

<br>

## **1. Introduction**

Drone detection poses unique challenges due to factors like small object size, rapid motion, and varying visual appearances under different environmental conditions. YOLO, introduced in 2015 by Redmon et al., revolutionized object detection by enabling single-pass, real-time inference. YOLOv11x, unveiled at YOLO Vision 2024, builds on this legacy with enhanced feature extraction, efficient backbone-neck design, and advanced attention mechanisms.

This project employs YOLOv11x’s extra-large configuration to achieve maximum accuracy for drone detection, supplemented with tracking and heatmap visualization capabilities.

<br>

## **2. Data Preprocessing**

The dataset was split into two subsets:

-   **Training set**: 1,012 images with 1,012 YOLO-format annotation TXT files.
    
-   **Validation set**: 347 images with 348 annotation TXT files.
    

Each annotation file used YOLO’s normalized format, containing the class ID and bounding box coordinates.

A `data.yaml` configuration file defined the dataset structure:

```yaml
train: ../drone_dataset/train
val: ../drone_dataset/valid
nc: 1
names: ['drone']

```

Images were resized to **960×960 pixels** with letterboxing to preserve aspect ratios. Data augmentation included mosaic augmentation, random scaling, and HSV color jitter to improve generalization.

<br>

## **3. Training**

Training was conducted on Google Colab Pro+ with **Torch 2.6.0+cu124** and NVIDIA L4 GPU acceleration.

Command:

```bash
!yolo train \
    model=yolov11x.pt \
    data=/content/data.yaml \
    imgsz=960 \
    epochs=32 \
    batch=16 \
    device=0

```

**Parameter breakdown:**

-   `model=yolov11x.pt`: Pre-trained YOLOv11x weights.
    
-   `data`: Path to dataset configuration.
    
-   `imgsz=960`: Higher resolution for better small-object detection.
    
-   `epochs=32`: Training cycles.
    
-   `batch=16`: Images per batch.
    
-   `device=0`: GPU device index.
    

**Training Summary:**

-   **Model**: YOLOv11x (fused) — 190 layers, 56,828,179 parameters, 194.4 GFLOPs.
    
-   **mAP50**: 0.905, **mAP50-95**: 0.546.
    
-   **Speed**: ~8.9 ms inference per image.
    

Results saved to `runs/detect/train`.

<br>

## **4. Inference and Tracking**

To perform detection and tracking:

```python
model = YOLO("/content/runs/detect/train/weights/best.pt")
result = model.track(
    source="/content/video.mp4",
    conf=0.3,
    iou=0.5,
    show=True
)

```

-   **`conf=0.3`**: Confidence threshold.
    
-   **`iou=0.5`**: IoU threshold for NMS.
    
-   **`show=True`**: Displays annotated frames.
    
<video width="600" controls>
  <source src="https://raw.githubusercontent.com/doguilmak/Drone-Detection-YOLOv7/main/detect/drone_detect_YOLOv7x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Drone detection prediction using YOLOv7x. Video footage by Nino Souza, sourced from Pexels.*

<br>

## **5. Heatmap Visualization**

To visualize spatial activity density:

```python
cap = cv2.VideoCapture("/content/pexels-joseph-redfield-8459631 (1080p).mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter("/content/my_outputs/tracking_output/heatmap_output.mp4", fourcc, fps,  (width, height))

heatmap = solutions.Heatmap(
	colormap=cv2.COLORMAP_PARULA,
	show=False,
	model="/content/runs/detect/train/weights/best.pt")

```

-   Heatmaps highlight areas of frequent drone presence.
    
-   `yolo11n.pt` ensures fast processing for visualization.
    

<video width="600" controls>
  <source src="https://raw.githubusercontent.com/doguilmak/Drone-Detection-YOLOv7/main/detect/drone_detect_YOLOv7x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Drone detection prediction using YOLOv7x. Video footage by Nino Souza, sourced from Pexels.*

<br>

## **6. Results**

**Validation Set Performance (347 images / 369 instances):**

-   **Precision**: 0.922
    
-   **Recall**: 0.831
    
-   **mAP@50**: 0.905
    
-   **mAP@50-95**: 0.546
    

**Speed:**

-   Preprocessing: 0.1 ms/image
    
-   Inference: 8.9 ms/image
    
-   Postprocessing: 1.0 ms/image
    
<br>

## **7. Requirements**

-   Python 3.8+
    
-   PyTorch 2.6.0+cu124
    
-   OpenCV
    
-   Ultralytics YOLOv11
    
-   CUDA-compatible GPU
    
<br>


### Drone Detection with YOLOv8x and YOLOv7

In addition to this work, I have also developed a drone detection model using YOLOv8x and YOLOv7.   

 - You can find that project or repository of YOLOv8x [here](https://github.com/doguilmak/Drone-Detection-YOLOv8x). 

 - You can find that project or repository of YOLOv7 [here](https://github.com/doguilmak/Drone-Detection-YOLOv7).

<br>

## **8. References**

-   Khanam, R., & Hussain, M. _YOLOv11: An Overview of the Key Architectural Enhancements_, 2024.
    
-   [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)

    

