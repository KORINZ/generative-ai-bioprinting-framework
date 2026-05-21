# Disentangled Generative AI-Guided Closed-Loop Optimization of Deposition Morphology for 3D Bioprinting Applications

This repository contains the main scripts used for the manuscript "Generative AI-guided in silico closed-loop optimisation of deposition morphology for 3D bioprinting applications."

## Abstract
Recent advances in novel bioinks have dramatically increased the feasibility and applicability of 3D bioprinting for tissue engineering and regenerative medicine. However, developing new bioinks still requires extensive trial-and-error testing due to bioink rheology, crosslinking reactions, printing parameters, and limited resources. Previous classification- or regression-based AI models for bioink optimisation are typically black-box and cannot provide visual results. To address these challenges, a state-of-the-art disentangled and explainable generative AI framework was developed. The framework comprises a beta-conditional variational autoencoder (β-CVAE) for generating novel, variational images of printed constructs based on ink properties and printing parameters. Furthermore, an in silico closed-loop Bayesian optimisation (BO) system coupled with a convolutional neural network (CNN) was employed to quantitatively predict pre-printing performance and classify defects. The trained β-CVAE model can generate realistic and condition-dependent images of printed constructs. Visualisation of the latent space revealed an interpretable organisation of the learned features, supporting the model’s explainability and controllability. Moreover, transfer learning was employed to rapidly adapt to new blueprint designs with limited data. Although this study focuses on acellular hydrogel printing, the bioink formulations and crosslinking conditions are cytocompatible and extensible to bioprinting applications. The proposed framework accelerates bioprinting optimisation through interpretable generative AI modelling.

**Keywords:** 
3D bioprinting; deposition morphology; machine learning; generative artificial intelligence; variational autoencoder

## GUI Screenshot
<p align="center">
  <a href="https://github.com/KORINZ/nhk-news-scraper-gui#nhk-news-web-easy-scraper">
    <img width="800" alt="GUI_white" src="https://github.com/user-attachments/assets/719a0225-1716-4bd8-9b85-b88c90c79557" />
  </a>
</p>

## Data Availability
Please note that some scripts in this repository may require the corresponding data files or models to run successfully. Some scripts may require changes to file paths to access the data.
The data that support the findings of this study are openly available in Zenodo at [https://doi.org/10.5281/zenodo.19602891](https://doi.org/10.5281/zenodo.19602891) (v0.0.2).

## Disclaimer
This repository is intended for educational and research purposes. The authors are not responsible for any misuse of the code or data provided herein. Users are encouraged to cite the original manuscript when using this code in their research.
