
# SPAC: Spectrum-informed Profiled Adaptive Calibration

SPAC is a calibration framework for car-following (CF) and fuel consumption models. It aims to identify competitive compromises between spacing and fuel-rate accuracy under limited online simulation budgets.

## Overview

The framework combines:

- **Fuel-parameter profiling:** Bounded least-squares estimation of fuel-model parameters.
- **Learned parameter prior:** A neural network conditioned on operating-state and spectral descriptors.
- **Preference-decomposed D-CEM:** Multi-objective calibration across predefined preferences.
- **Spectral constraints:** Candidate screening and acceleration-spectrum admissibility checks.

## Supported Models

Car-following models:
- Intelligent Driver Model (IDM)
- Gipps model
- Gazis-Herman-Rothery (GHR) model
- Optimal Velocity Model (OVM)

Fuel consumption model:
- VT-CPFM-1

## Supported Dataset
[Arizona Ring Experimental Dataset (ARED)](https://vanderbilt.app.box.com/s/1oe1g417wtkzkjgtw5y2c1vdx5gcqm85)

## Installation

Installation instructions will be provided with the source code.

## Usage

Configuration details and reproducibility instructions will be provided with the implementation.

## Citation

Citation information will be updated upon publication.

## License

See [LICENSE](LICENSE) for details.
