# py_image_similarity_search
Python image knn similarity search

## Introduction
This repository implements KNN image similarity search
This workflow is split into two parts
1. Hash Generation

   Since the dimensionality of image is huge, thus we need to hash it to samller dimension
2. KNN search

   After we got hash of image, I use KNN for find K similar images

# Usage
Generally as bellow, however some modification is required for configuring INPUT/OUTPUT directory.
1. Hash Generation

   `python main.py`
2. KNN search

   `python query.py`
