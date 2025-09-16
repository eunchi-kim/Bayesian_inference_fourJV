# Parameter estimation from current-voltage curves 

![Graphical abstract](graphical%20abstract.png)

This repository contains code for parameter estimation from current-voltage curves with a NN-based surrogate model.

> **Note:**  
> Due to licensing issues for the drift-diffusion simulator, the corresponding code is omitted.

For details, see the main paper  	
https://doi.org/10.48550/arXiv.2506.13308: 

---

## 📁 Example Data

Large example datasets (>1GB) used in this project are hosted on Zenodo due to GitHub size limits.

**Download data:**  
[https://zenodo.org/record/15480770](10.5281/zenodo.15480769)  

**Description:**
- `exp_data_example_d171nm_4illus_0p02-0p84V.h5`: Raw illumination-dependent JV curves (active blend - T1:BTP-4F-12)
- `example_d171nm_8_param_train_test.h5`: Train and test dataset for building the NN model
- `example_d171nm_8_paramscaler.joblib`: Standardization tool for NN model input (material parameters)
- `y1/example_d171nm_8_param_y1_trained_model.h5`: Trained NN model for shifted JV curves
- `y2/example_d171nm_8_param_y2_trained_model.h5`: Trained NN model for short-circuit current densities

**How to use:**
1. Download the required files from Zenodo.
2. Extract them into the repository folder.
3. Use the `pdf_analysis_example.ipynb` notebook to infer parameters from experimental or NN-generated data and visualize the results.


---

## Requirements

- Python 3.8+
- numpy, matplotlib, tensorflow, pymoo, ...

Install all dependencies using:
```bash
pip install -r requirements.txt
