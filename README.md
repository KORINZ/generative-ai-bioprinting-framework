# Disentangled Generative AI-Guided Closed-Loop Optimization of Deposition Morphology for 3D Bioprinting Applications

This repository contains the main scripts used for the manuscript "Generative AI-guided in silico closed-loop optimisation of deposition morphology for 3D bioprinting applications."

## Abstract
Recent advances in novel bioinks have dramatically increased the feasibility and applicability of 3D bioprinting for tissue engineering and regenerative medicine. However, developing new bioinks still requires extensive trial-and-error testing due to bioink rheology, crosslinking reactions, printing parameters, and limited resources. Previous classification- or regression-based AI models for bioink optimisation are typically black-box and cannot provide visual results. To address these challenges, a state-of-the-art disentangled and explainable generative AI framework was developed. The framework comprises a beta-conditional variational autoencoder (β-CVAE) for generating novel, variational images of printed constructs based on ink properties and printing parameters. Furthermore, an in silico closed-loop Bayesian optimisation (BO) system coupled with a convolutional neural network (CNN) was employed to quantitatively predict pre-printing performance and classify defects. The trained β-CVAE model can generate realistic and condition-dependent images of printed constructs. Visualisation of the latent space revealed an interpretable organisation of the learned features, supporting the model’s explainability and controllability. Moreover, transfer learning was employed to rapidly adapt to new blueprint designs with limited data. Although this study focuses on acellular hydrogel printing, the bioink formulations and crosslinking conditions are cytocompatible and extensible to bioprinting applications. The proposed framework accelerates bioprinting optimisation through interpretable generative AI modelling.

**Keywords:** 
3D bioprinting; deposition morphology; machine learning; generative artificial intelligence; variational autoencoder

<a href="https://www.tandfonline.com/doi/full/10.1080/17452759.2026.2671497">[Open Access Article Link]</a>

## Graphical Abstract
<p align="center">
  <a href="https://doi.org/10.1080/17452759.2026.2671497">
    <img width="1000" alt="GraphicalAbstract3" src="https://github.com/user-attachments/assets/0512985f-ca30-49cc-8373-7b47a0fe7677" />
  </a>
  <br>
  <sub>Adapted from Zhang, C. et al., <i>Virtual and Physical Prototyping</i> <b>21</b>, e2671497 (2026), under CC BY 4.0.</sub>
</p>


## GUI Animation
<p align="center">
  <a href="https://doi.org/10.1080/17452759.2026.2671497">
    <img width="1280" height="931" alt="GUI Animation" src="https://github.com/user-attachments/assets/b225e755-f2d4-480b-8b66-77dd1e6012ee" />
  </a>
</p>

## Citation
Colin Zhang, Kelum Elvitigala, and Shinji Sakai.
Generative AI-guided in silico closed-loop optimisation of deposition morphology for 3D bioprinting applications. <i>Virtual and Physical Prototyping</i> <b>21</b>, e2671497 (2026). <a href="https://doi.org/10.1080/17452759.2026.2671497">https://doi.org/10.1080/17452759.2026.2671497</a>.

```bibtex
@article{zhang_genai_2026,
  author = {Colin Zhang and Kelum Elvitigala and Shinji Sakai},
  title = {Generative AI-guided in silico closed-loop optimisation of deposition morphology for 3D bioprinting applications},
  journal = {Virtual and Physical Prototyping},
  volume = {21},
  number = {1},
  pages = {e2671497},
  year = {2026},
  month = {may},
  publisher = {Taylor \& Francis},
  doi = {10.1080/17452759.2026.2671497},
  url = {https://doi.org/10.1080/17452759.2026.2671497}
}
```

## Data Availability
Please note that some scripts in this repository may require the corresponding data files or models to run successfully. Some scripts may require changes to file paths to access the data.
The data that support the findings of this study are openly available in Zenodo at [https://doi.org/10.5281/zenodo.19602891](https://doi.org/10.5281/zenodo.19602891) (v0.0.2).

## Disclaimer
Please note that some scripts in this repository might require the corresponding data files to run successfully. This repository is intended for educational and research purposes. The authors are not responsible for any misuse of the code or data provided herein. Users are encouraged to cite the original manuscript when using this code in their research.

