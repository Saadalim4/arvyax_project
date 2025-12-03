# Hand Tracking Virtual Boundary System

A real-time hand tracking system that detects when a user's hand approaches a virtual object and triggers visual warnings. Built using classical computer vision techniques without external pose-detection APIs.

## 🎯 Features

- **Real-time Hand Tracking**: Uses HSV color segmentation and contour detection
- **Virtual Boundary Zones**: Three concentric zones (SAFE/WARNING/DANGER)
- **Dynamic State Logic**: Distance-based classification with visual feedback
- **CPU-optimized**: Runs at 15-25 FPS on standard laptops
- **No External APIs**: Pure OpenCV + NumPy implementation

## 🖥️ Demo

![Demo GIF](demo/demo.gif) *Add your demo video here*

**States:**
- **SAFE** (Green): Hand far from virtual object
- **WARNING** (Yellow): Hand approaching virtual object  
- **DANGER** (Red): Hand touching/close to boundary with "DANGER DANGER" warning

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/hand-tracking-boundary.git
cd hand-tracking-boundary

# Install dependencies
pip install -r requirements.txt