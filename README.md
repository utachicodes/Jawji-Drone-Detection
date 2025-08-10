<h1 align=center><font size = 6>Drone Detection and Tracking using YOLOv11x</font></h1>

<img src="https://images.pexels.com/photos/724921/pexels-photo-724921.jpeg" height=550 width=1000 alt="https://www.pexels.com/"/>

<small>Picture Source: <a href="https://www.pexels.com/@pok-rie-33563/">Pok Rie</a></small>

<br>

## **Abstract**

Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used in sectors ranging from surveillance and delivery to agriculture. While offering significant benefits, their proliferation also raises concerns about security, privacy, and airspace safety. This project presents a robust, real-time drone detection and tracking system based on **YOLOv11x**, the latest iteration of the YOLO (You Only Look Once) object detection family. Leveraging architectural innovations like **C3k2 blocks**, **SPPF**, and **C2PSA spatial attention**, the model excels at detecting small, fast-moving aerial targets in diverse and complex environments.

### **Requirements**

-   Python 3.8+
    
-   PyTorch 2.6.0+cu124
    
-   OpenCV
    
-   Ultralytics YOLOv11
    
-   CUDA-compatible GPU

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

Images were resized to **640×640 pixels** with letterboxing to preserve aspect ratios. Data augmentation included mosaic augmentation, random scaling, and HSV color jitter to improve generalization.

<br>

## **3. Training**

Training was conducted on Google Colab Pro+ with **Torch 2.6.0+cu124** and NVIDIA L4 GPU acceleration.

Command:

```python
results = model.train(
        batch=16,
        data="data.yaml",
        epochs=32,
        imgsz=640,
        lr0=0.001,
        optimizer='AdamW'
    )
```

**Parameter breakdown:**

-   `model=yolov11x.pt`: Pre-trained YOLOv11x weights.
    
-   `data`: Path to dataset configuration.
    
-   `imgsz=640`: Higher resolution for better small-object detection.
    
-   `epochs=32`: Training cycles.
    
-   `batch=16`: Images per batch.
    
-   `device=0`: GPU device index.
-   `lr0=0.001`: Learing rate.
-   `optimizer='AdamW'` defined training optimization algorithm.
    

**Training Summary:**

-   **Model**: YOLOv11x (fused) — 190 layers, 56,828,179 parameters, 194.4 GFLOPs.
    
-   **mAP50**: 0.905, **mAP50-95**: 0.546.
    
-   **Speed**: ~8.9 ms inference per image.
    

Results saved to `runs/detect/train`.

<br>

<br>

## **4. Results**

We present comprehensive results of our drone detection model's performance on both the training and testing datasets. The evaluation metrics include precision, recall, and F1-score, which are standard measures to assess the model's detection accuracy. Additionally, we analyze the model's performance across various environmental conditions and discuss its strengths and limitations.

<img  src="https://github.com/doguilmak/Drone-Detection-YOLOv11x/blob/main/assets/results.png" width=1000  height=400 alt="github.com/doguilmak/"/>

**Validation Set Performance (347 images / 369 instances):**

-   **Precision**: 0.922
    
-   **Recall**: 0.831
    
-   **mAP@50**: 0.905
    
-   **mAP@50-95**: 0.546
    
**Speed**

The trained YOLOv11x model for drone detection demonstrates impressive processing efficiency across all stages of the detection pipeline. During preprocessing, the model requires only 0.1 milliseconds per image, indicating a highly optimized input handling mechanism that allows for rapid normalization and resizing of images prior to inference. The inference stage, which involves the core computation and prediction processes of the deep neural network, completes in approximately 8.9 milliseconds per image. This reflects the model's capacity to deliver fast object detection results while maintaining the high computational demands of YOLOv11x. Following inference, the postprocessing step, including operations such as non-maximum suppression and bounding box formatting, is completed in just 1.0 millisecond per image.

<br>

## **5. Inference and Tracking**

After training, the YOLOv11x model was used for real-time detection and tracking of drones in video streams. The tracking module integrates YOLO detection outputs with a multi-object tracking (MOT) algorithm, such as ByteTrack or DeepSORT, to maintain object identities across frames, even under partial occlusion or rapid movement. To perform detection and tracking:

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

<br>

<p align="center">
  <img src="assets/pexels-joseph-redfield-8459631 (1080p).gif" width="800" alt="github.com/doguilmak/"/>
</p>

<p align="center">
  <em>Drone detection prediction using YOLOv11x. Video footage by Nino Souza, sourced from Pexels.</em>
</p>

<br>

## **6. Heatmap Visualization**

To derive deeper insights into spatial and temporal patterns of drone activity, heatmap visualization is integrated into the analysis pipeline. This technique effectively aggregates drone detections across video frames, producing a visual representation of the frequency and concentration of drone occurrences in different regions of the frame. Areas with consistently high detection density are rendered with warmer colors, making it easy to identify operational hotspots, typical flight corridors, or zones of repeated drone activity—crucial for surveillance. In the implementation below, the system reads a video stream, tracks drone positions over time using the trained YOLOv11x model, and overlays a heatmap to reflect cumulative activity density:

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

<br>
 
<p align="center">
  <img src="assets/heatmap_output.gif" width="800" alt="github.com/doguilmak/"/>
</p>

<p align="center">
  <em>Drone detection prediction using YOLOv11x. Video footage by Nino Souza, sourced from Pexels.</em>
</p>

<br>

This visualization not only supports post-hoc analysis of drone flight behavior but also enhances situational awareness in live monitoring systems by revealing zones of concentrated aerial activity.
 
<br>

## **8. References**

Khanam, R., & Hussain, M. _YOLOv11: An Overview of the Key Architectural Enhancements_, 2024.
    
[Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)

<br>

## Drone Detection with YOLOv11x

In addition to this work, I have also developed drone detection models using YOLOv8x and YOLOv7.

- You can find the YOLOv8x project or repository [here](https://github.com/doguilmak/Drone-Detection-YOLOv8x).

- You can find the YOLOv7 project or repository [here](https://github.com/doguilmak/Drone-Detection-YOLOv7).

<br>
