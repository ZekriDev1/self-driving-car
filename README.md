# Self-Driving Car Simulation 🚗🤖

A desktop simulation where multiple autonomous cars learn to navigate a track using neural networks and a genetic algorithm. Each car evolves independently, improving performance over generations.

---

## Features

- **Independent Cars**: Each car has its own neural network. Crashes do not affect other cars.  
- **Sensors**: Cars detect track boundaries and obstacles using raycasting and other inputs.  
- **Neural Network Control**: Cars handle steering and braking autonomously.  
- **Genetic Algorithm**: Evolution via selection, elitism, and mutation improves performance over generations.  
- **Fitness Evaluation**: Cars are scored on distance, stability, and path adherence.  
- **Crash Handling**: Collided cars are removed instantly from the simulation.  

---

## Tech Stack

- **Python**  
- **NumPy**  
- **Pygame**  
- Custom Neural Networks  
- Genetic Algorithms  

---

## Project Structure
self_driving_cars/
├─ main.py # Main simulation loop
├─ car.py # Car class and movement logic
├─ neural_network.py # Neural network implementation
├─ genetic_algorithm.py # Evolutionary algorithm logic
├─ sensors.py # Sensor handling (raycasting)
├─ config.py # Simulation and car parameters
├─ track.py # Track generation and layout
├─ renderer.py # Visualization
├─ best_brain.json # Saved best-performing network
└─ pycache/ # Python cache files (ignored in Git)


---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/ZekriDev1/self-driving-car.git
cd self-driving-car

2. Install dependencies:

pip install numpy pygame
