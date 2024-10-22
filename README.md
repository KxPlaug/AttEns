

# Attribution Algorithm Ensemble (AttEns)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/KxPlaug/AttEns)

## Introduction

This repository provides the official implementation of the paper **"Optimizing Attribution with Ensemble Methods for Improved Model Interpretability"** under reviewing at ICASSP 2025. 

The *Attribution Algorithm Ensemble (AttEns)* framework integrates multiple attribution methods to improve the interpretability of machine learning models while maintaining computational efficiency. The framework has been tested on models such as Inception-v3, ResNet-50, VGG16, and MaxViT-T, demonstrating superior performance in insertion and deletion metrics.

## Key Features

- **Ensemble of Attribution Methods**: Combines multiple state-of-the-art attribution algorithms (IG, SG, BIG, etc.) to improve the quality of model explanations.
- **Optimized Computational Complexity**: Reduces the computational burden by minimizing the number of forward and backward propagations.
- **Extensive Experimentation**: Experiments across various models and datasets demonstrate the advantages of the ensemble approach.
- **Extensible Framework**: Easily integrates new attribution methods for broader research exploration.



## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- jupyter (for running notebooks)

To install the necessary dependencies, you can use:
```bash
pip install -r requirements.txt
```

### Dataset

We use the [ImageNet](https://www.image-net.org/) dataset for the experiments in our paper. Ensure that you have downloaded and preprocessed the dataset before running the experiments.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KxPlaug/AttEns.git
   cd AttEns
   ```

2. **Run all experiments**:
   The `run_all.py` script will initiate all necessary attribution methods on the chosen model and dataset.
   ```bash
   python run_all.py
   ```

3. **Evaluate initial attribution results**:
   After running the attribution algorithms, use the `eval_all.py` script to evaluate the results and compute initial metrics.
   ```bash
   python eval_all.py
   ```

4. **Combine attribution results**:
   Open the Jupyter notebook `choice.ipynb` to combine the results from the multiple attribution methods. This step allows you to manually or automatically choose the best attributions from the ensemble.
   ```bash
   jupyter notebook choice.ipynb
   ```

5. **Evaluate combined attribution results**:
   After combining the attribution results, run the `eval_all_ori.py` script to compute the final evaluation metrics based on the ensemble method.
   ```bash
   python eval_all_ori.py
   ```


## Results

The experimental results in our paper demonstrate the superior performance of the AttEns framework compared to baseline attribution algorithms. Below are the key results:

|     Method    	| Inception-v3 	|          	| ResNet-50 	|          	|   VGG16  	|          	| MaxViT-T 	|          	| Number of Propagation 	|          	|
|:-------------:	|:------------:	|:--------:	|:---------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:---------------------:	|:--------:	|
|               	|      INS     	|    DEL   	|    INS    	|    DEL   	|    INS   	|    DEL   	|    INS   	|    DEL   	|        Forward        	| Backward 	|
|      FIG      	|   0.201722   	| 0.045629 	|  0.106266 	| 0.032358 	|  0.07933 	| 0.027029 	| 0.462121 	| 0.182173 	|           1           	|     1    	|
|    DeepLIFT   	|   0.295152   	| 0.041533 	|  0.124774 	| 0.030503 	|  0.09342 	| 0.023026 	| 0.498488 	| 0.181546 	|           1           	|     1    	|
|      GIG      	|   0.318705   	| 0.034337 	|  0.144963 	| 0.019132 	|  0.10252 	| 0.017318 	| 0.545019 	| 0.138792 	|           15          	|    15    	|
|       IG      	|   0.320825   	|  0.04258 	|  0.145412 	| 0.028333 	| 0.095863 	| 0.023163 	| 0.541594 	|  0.18818 	|           1           	|     1    	|
|       EG      	|   0.375499   	| 0.265265 	|  0.350081 	|  0.28264 	| 0.356653 	| 0.337217 	| 0.598585 	| 0.515332 	|           1           	|     1    	|
|       SG      	|    0.38911   	| 0.033278 	|  0.277219 	| 0.022857 	| 0.186208 	| 0.016392 	| 0.641634 	| 0.139467 	|           1           	|     1    	|
|      BIG      	|    0.48401   	| 0.053815 	|  0.290461 	| 0.046713 	| 0.226557 	| 0.037233 	| 0.568201 	| 0.186599 	|          359          	|    351   	|
|       SM      	|   0.533356   	|  0.0631  	|  0.31544  	| 0.056741 	| 0.270308 	| 0.041743 	| 0.489565 	| 0.195568 	|           1           	|     1    	|
|     MFABA     	|   0.538468   	| 0.063881 	|  0.320002 	| 0.055452 	| 0.279122 	| 0.040634 	| 0.440624 	| 0.358368 	|           51          	|    51    	|
|      AGI      	|   0.572294   	| 0.058431 	|  0.500747 	| 0.051438 	| 0.397331 	| 0.042029 	| 0.645392 	| 0.198408 	|          401          	|    800   	|
|   AttEXplore  	|   0.618792   	| 0.044244 	|  0.504209 	| 0.033338 	| 0.442779 	|  0.0282  	| 0.615814 	|  0.15773 	|          221          	|    220   	|
|       LA      	|   0.646301   	| 0.067499 	|  0.549666 	| 0.047156 	| 0.437295 	|  0.03378 	| 0.704771 	| 0.208705 	|          602          	|   1170   	|
| AttEns (Ours) 	|   0.735599   	| 0.033247 	|  0.610617 	| 0.029891 	| 0.521829 	| 0.028776 	| 0.728149 	| 0.117611 	|          229          	|    236   	|

For more details on the experiments, refer to our paper or the [results](https://github.com/KxPlaug/AttEns/results) folder.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

**Citation**:
```

```