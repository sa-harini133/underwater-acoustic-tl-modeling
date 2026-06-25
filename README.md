# underwater-acoustic-tl-modeling
Bellhop-integrated USBL communication system for underwater acoustic propagation analysis and real-time channel visualization.
# NIOT USBL + Bellhop Integration

## Overview

This project integrates the **Bellhop underwater acoustic propagation model** into an indigenous **USBL (Ultra Short Baseline) modem graphical interface**, enabling real-time visualization and analysis of underwater acoustic communication channels.

The application combines acoustic propagation modeling with modem communication to estimate transmission loss, ray paths, sound speed profiles, and environmental effects for both shallow and deep water deployments.

---

## Features

* Integration of Bellhop acoustic propagation model with indigenous USBL GUI
* Support for both **Shallow Water** and **Deep Water** environments
* Automatic Bellhop environment (.env) file generation
* Ray tracing visualization
* Transmission Loss (TL) heatmap generation
* Sound Speed Profile (SSP) visualization
* End-to-end Bellhop workflow

  * Environment generation
  * Ray tracing
  * Coherent transmission loss
  * Eigenray arrival analysis
* Python-based transmission loss estimation when Bellhop is unavailable
* Real-time acoustic channel analysis
* Integrated USBL modem communication interface
* Environmental parameter configuration
* Acoustic loss budget computation

---

## System Architecture

```
           USBL Modem
                │
                ▼
      Indigenous GUI Interface
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 Bellhop Integration   Acoustic Models
        │                │
        └───────┬────────┘
                ▼
      Transmission Loss Analysis
                │
                ▼
      Visualization Dashboard
```

---

## Bellhop Workflow

The application automatically performs:

1. Environment (.env) generation
2. Ray tracing simulation
3. Coherent transmission loss computation
4. Eigenray arrival analysis
5. Visualization of generated outputs

Supported Bellhop output files:

* `.env`
* `.prt`
* `.ray`
* `.shd`
* `.arr`

---

## Acoustic Models Implemented

The software incorporates multiple underwater propagation models, including:

* Geometric spreading
* Francois–Garrison absorption
* Thorp absorption model
* Surface scattering
* Bottom reflection losses
* Volume scattering
* Bubble attenuation
* Ambient noise (Wenz curves)
* Doppler shift estimation
* Sound Speed Profile generation

---

## Environment Support

### Shallow Water

* Freshwater pond environment
* Custom sound speed profile
* Mud bottom model
* Short-range communication

### Deep Water

* Coastal ocean environment
* Thermocline-based sound speed profile
* Sandy bottom model
* Long-range propagation

---

## Visualization

The GUI provides:

* Transmission Loss Heatmaps
* Ray Diagrams
* Sound Speed Profiles
* Acoustic Loss Breakdown
* Bellhop Output Viewer
* Live USBL Communication Status

---

## Technologies Used

* Python
* Tkinter
* NumPy
* Matplotlib
* Bellhop
* MAVLink
* Socket Programming
* Threading
* Scientific Computing

---

## Applications

* Underwater acoustic communication
* USBL modem analysis
* Marine robotics
* Autonomous Underwater Vehicles (AUVs)
* Oceanographic research
* Acoustic propagation studies
* Underwater localization
* Channel performance evaluation

---

## Future Improvements

* Real-time hardware-in-the-loop testing
* Integration with additional propagation models
* 3D acoustic propagation visualization
* Support for multiple simultaneous modems
* Performance optimization using GPU acceleration

---

## Acknowledgements

Developed as part of underwater acoustic communication research involving the integration of Bellhop with an indigenous USBL communication interface for underwater channel analysis and visualization.
