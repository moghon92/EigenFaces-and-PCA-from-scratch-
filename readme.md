# PCA from Scratch and Eigen Faces Generation

This project demonstrates the implementation of Principal Component Analysis (PCA) from scratch and utilizes it to generate eigenfaces. PCA is a popular dimensionality reduction technique used for extracting important features and patterns from high-dimensional data.

## Contents

- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Eigen Faces Generation](#eigen-faces-generation)

## Overview

PCA aims to find a lower-dimensional representation of the data while preserving the most significant information. It achieves this by projecting the data onto a new orthogonal basis, called the principal components, which are ordered in terms of the amount of variance they capture. By selecting a subset of the principal components, we can effectively reduce the dimensionality of the data.

This project includes two main scripts:
- `PCA.py`: Implements PCA from scratch, including the computation of eigenvalues and eigenvectors.
- `EigenFaces.py`: Uses the PCA implementation to generate eigen faces from a dataset of facial images.

## Implementation Details

The project consists of the following files:

- `PCA.py`: Contains the implementation of PCA from scratch, including the calculation of eigenvalues, eigenvectors, and variance explained.
- `EigenFaces.py`: Utilizes the PCA implementation to generate eigen faces from a dataset of facial images.


Please refer to the provided scripts for additional usage details and examples.

## Eigen Faces Generation

The following image demonstrates the process of generating eigen faces using PCA:

![Eigen Faces Generation](https://mikedusenberry.com/assets/images/faces_original.jpg)

This image provides a visual representation of the eigen faces obtained through the PCA algorithm.
