# Computer Vision Projects

This repository contains three different computer vision projects I worked on after successfully completing courses from Murtaza Hassan, an educator at the CVZone website. The projects apply object detection and tracking using YOLO and SORT algorithms to real-world problems. While the idea for these projects comes from the course, my goal was to practice and apply the knowledge I gained in computer vision to practical use cases.

## Table of Contents
1. [Car Counter](#car-counter)
2. [People Counter](#people-counter)
3. [PPE Detection](#ppe-detection)
4. [Technologies Used](#technologies-used)
5. [Installation and Setup](#installation-and-setup)
6. [Acknowledgements](#acknowledgements)

---

## Car Counter

This project detects and counts cars, trucks, buses, and motorbikes from a video feed. It uses the YOLOv8 model for detecting objects and SORT (Simple Online and Realtime Tracking) for tracking objects across frames. A pre-defined region of interest (ROI) is masked to focus detection on relevant areas. Once an object crosses the defined line, it is counted.

### Features
- Detects vehicles such as cars, trucks, buses, and motorbikes.
- Uses a mask to focus the detection on a specific region.
- Counts vehicles crossing a designated line.

### Files
- `car_counter.py`: Main Python script for car counting.
- `input-data/cars.mp4`: The input video file.
- `input-data/mask.png`: Mask image for filtering the detection region.

---

## People Counter

This project tracks and counts people in a video feed. Similar to the car counter, it uses YOLOv8 for detecting objects and SORT for tracking. The system counts people moving across two different boundaries: one for upward movement and another for downward movement.

### Features
- Detects and tracks people in a video.
- Counts people moving in two directions: up and down.

### Files
- `people_counter.py`: Main Python script for counting people.
- `input-data/people.mp4`: The input video file.
- `input-data/mask-people.png`: Mask image for filtering the detection region.

---

## PPE Detection

This project detects personal protective equipment (PPE) such as hardhats, safety vests, and masks in a workplace environment. It identifies both the presence and absence of safety equipment and provides visual feedback for safety compliance.

### Features
- Detects various safety equipment: hardhats, masks, and safety vests.
- Identifies violations when PPE is missing (e.g., NO-Hardhat, NO-Safety Vest, NO-Mask).
- Differentiates between compliant and non-compliant workers using color-coded bounding boxes.

### Files
- `ppe_detection.py`: Main Python script for PPE detection.
- `input-data/ppe-1-1.mp4`: The input video file.

---

## Technologies Used

- **YOLOv8**: For object detection.
- **SORT**: For object tracking.
- **OpenCV**: For image and video processing.
- **CVZone**: For drawing utilities (corner rectangles, text overlays, etc.).
- **NumPy**: For numerical computations.

---

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/computer-vision-projects.git
   cd computer-vision-projects
   ```