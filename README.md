## Project Overview

This project focuses on developing a model to detect dog noses using the YOLOv8 object detection framework. The aim is to create an accurate and reliable detection system that can be used for scientific research purposes in the future.

![image](https://github.com/user-attachments/assets/319db805-c043-45a6-98f5-e1959eb3e7d2)


## Prerequisites

Ensure you have the following libraries installed:

- `ultralytics`
- `roboflow`
- `supervision`
- `opencv-python`

You can install these libraries using pip:
```bash
pip install ultralytics roboflow supervision opencv-python
```

## Project Structure

The project consists of the following main components:

1. **Dataset Preparation**: Downloading and preparing the dataset from Roboflow.
2. **Model Training**: Training the YOLOv8 model on the dog nose dataset.
3. **Model Evaluation**: Evaluating the trained model's performance.
4. **Inference and Visualization**: Running the model on a sample image and visualizing the detected dog noses.

## Setup and Usage

### 1. Environment Setup

Clone the repository and navigate to the project directory. Ensure your current working directory is the project directory:
```python
import os
HOME = os.getcwd()
print(HOME)
```

### 2. Install Required Libraries

Install the necessary libraries:
```bash
pip install ultralytics --quiet
```

### 3. Download and Prepare the Dataset

Use Roboflow to download the dataset:
```python
import roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("school-2bcb4").project("dog-nose-0geek")
version = project.version(1)
dataset = version.download("yolov8")
```

### 4. Train the Model

Load a pretrained YOLOv8 model and train it on the dataset:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pretrained model

# Train the model
results = model.train(data='/content/dog-nose-1/data.yaml', epochs=85)
```

### 5. Evaluate the Model

Evaluate the trained model to check its performance:
```python
model = YOLO('/content/runs/detect/train9/weights/best.pt')
metrics = model.val()
```

### 6. Run Inference and Visualize Results

Perform inference on a sample image and visualize the detected dog noses:
```python
import supervision as sv
import cv2

image = 'dog.jpeg'
results = model.predict(image)

detections = sv.Detections.from_ultralytics(results[0])
image = cv2.imread(image)

bounding_box_annotator = sv.BoundingBoxAnnotator()
classes = model.names

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
sv.plot_image(annotated_image)
```

### 7. Save the Model Outputs

Zip the model output files for future use:
```python
%cd /content
import locale
locale.getpreferredencoding = lambda: "UTF-8"
!zip -r /content/runs.zip /content/runs
```

## Acknowledgments

- The YOLOv8 implementation by Ultralytics.
- Roboflow for providing the dataset and tools for dataset management.
