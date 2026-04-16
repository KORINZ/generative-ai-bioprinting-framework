# Disentangled Generative AI-Guided Closed-Loop Optimization of Deposition Morphology in 3D Bioprinting

This repository contains the main scripts used for the manuscript "Generative AI-guided in silico closed-loop optimization of deposition morphology in 3D bioprinting" (under review).

## Abstract
Recent advances in novel bioinks have dramatically increased the feasibility and applicability of 3D bioprinting for tissue engineering and regenerative medicine. However, developing new bioinks still requires extensive trial-and-error testing to achieve optimization. Several factors impede the optimization process, including bioink rheology, crosslinking reactions, printing parameters, and limited resources. Previous studies have used classification- or regression-based machine learning models for bioink optimization. Nevertheless, these models are typically black-box models and cannot provide visual results. To address these challenges, a state-of-the-art disentangled and explainable generative AI framework was developed. The framework comprises a beta-conditional variational autoencoder (β-CVAE) for generating novel, variational images of printed constructs based on ink properties and printing parameters. Furthermore, an in silico closed-loop Bayesian optimization (BO) system coupled with a convolutional neural network (CNN) was employed to quantitatively predict pre-printing performance and classify construct defects in generated images. The trained β-CVAE model can generate realistic and condition-dependent images of printed constructs. Visualization of the latent space revealed an interpretable organization of the learned features, supporting the model’s explainability and controllability. Moreover, transfer learning was employed to rapidly adapt to new blueprint designs with limited data. The proposed framework accelerates bioink optimization through interpretable generative AI modeling.

**Keywords:** 
3D bioprinting; deposition morphology; machine learning; generative artificial intelligence; variational autoencoder


## Data Availability
Please note that some scripts in this repository might require the corresponding data files to run successfully. The data that support the findings of this study are openly available in Zenodo at [https://doi.org/10.5281/zenodo.19602891](https://doi.org/10.5281/zenodo.19602891) (v0.0.2).

## Disclaimer
This repository is intended for educational and research purposes. The authors are not responsible for any misuse of the code or data provided herein. Users are encouraged to cite the original manuscript when using this code in their research.
