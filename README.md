# Tennis Game Analysis AI System

This project is an advanced AI system designed for analyzing tennis games. It leverages state-of-the-art computer vision techniques, including YOLOv8, YOLOv5, and ResNet, to recognize players, the ball, and the court. The system provides detailed insights into the speed of each player, their average speed throughout the game, the speed of their shots, and their average shot speed. It also features a mini court visualization to enhance the analysis. This project represents a significant step forward in applying AI to sports analytics, offering valuable data for coaches, players, and fans alike.

## Features

- **Player and Ball Recognition**: Utilizes YOLOv8 and YOLOv5 models for accurate detection and tracking of players and the ball.
- **Speed Analysis**: Calculates the speed of each player and their shots, along with their average speeds throughout the game.
- **Court Detection**: Identifies the tennis court using a custom model based on ResNet, ensuring accurate positioning and analysis.
- **Mini Court Visualization**: Provides a visual representation of the game, highlighting key moments and statistics.

https://github.com/Artin200912/Tennis-Game-Analysis/assets/136892986/80f6521e-8d20-4153-a045-29d9db439885


## Getting Started

### Prerequisites

- Python 3.6 or higher
- TensorFlow or PyTorch (depending on the YOLO version used)
- OpenCV for video processing

### Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/tennis-game-analysis.git
   ```
2. **Navigate to the project directory**:
   ```
   cd tennis-game-analysis
   ```
3. **Install the required packages**:
   ```
   pip install -r req.txt
   ```

### Usage

To run the analysis on a tennis game video, execute the following command in your terminal:
```
python main.py
```


## Contributing

Contributions to this project are welcome. Whether you're adding new features, improving the analysis algorithms, or fixing bugs, your contributions can help make this tennis game analysis AI system even more powerful. Please follow the standard GitHub workflow for contributing:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Push your branch to your fork.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The development of this tennis game analysis AI system is inspired by the advancements in computer vision and AI. We acknowledge the contributions of the YOLO, ResNet, and TensorFlow/PyTorch communities.
- Special thanks to all contributors who have helped make this project a reality.
