# Tennis Game Analysis AI System

This project is an advanced AI system designed for analyzing tennis games. It leverages state-of-the-art computer vision techniques, including YOLOv8, YOLOv5, and ResNet, to recognize players, the ball, and the court. The system provides detailed insights into the speed of each player, their average speed throughout the game, the speed of their shots, and their average shot speed. It also features a mini court visualization to enhance the analysis. This project represents a significant step forward in applying AI to sports analytics, offering valuable data for coaches, players, and fans alike.

## Features

- **Player and Ball Recognition**: Utilizes YOLOv8 and YOLOv5 models for accurate detection and tracking of players and the ball.
- **Speed Analysis**: Calculates the speed of each player and their shots, along with their average speeds throughout the game.
- **Court Detection**: Identifies the tennis court using a custom model based on ResNet, ensuring accurate positioning and analysis.
- **Mini Court Visualization**: Provides a visual representation of the game, highlighting key moments and statistics.

[Video Preview](https://github.com/Artin200912/Tennis-Game-Analysis/blob/main/output_videos/output_video.avi)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- TensorFlow or PyTorch (depending on the YOLO version used)
- OpenCV for video processing

### Installation

1. Clone the repository:
