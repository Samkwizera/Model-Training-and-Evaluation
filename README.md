# Cassava Leaf Disease Classification — ML vs Deep Learning

A comparative study of traditional machine learning (Scikit-learn) and deep learning
(TensorFlow) approaches for classifying cassava leaf diseases from field images.

**Summative project — Introduction to Machine Learning, African Leadership University.**

## Problem

Cassava is the second largest source of carbohydrates in sub-Saharan Africa and a
critical food-security crop. Four viral and bacterial diseases — Cassava Bacterial
Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), and
Cassava Mosaic Disease (CMD) — cause major yield losses across East Africa. Early,
accurate, low-cost disease identification from a smartphone photo can help
smallholder farmers act before losses spread.

This project frames the problem as a 5-class image classification task
(4 diseases + healthy) and compares two modelling families on the same data:
classical ML pipelines and convolutional neural networks.

## Dataset

[Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
(Kaggle, 2020), collected by the Makerere AI Lab from regular field surveys in
Uganda. 21,397 labelled images across 5 classes. The class distribution is heavily
imbalanced — CMD and CBSD together account for roughly 72% of the images, which
makes per-class evaluation and class-weighting non-trivial.

## Approaches

Two families of models are trained and evaluated on the same train / validation /
test split:

- **Classical ML (Scikit-learn).** Hand-crafted features (color histograms, HOG)
  and pretrained-CNN embeddings as input to Logistic Regression, Random Forest, and
  SVM.
- **Deep learning (TensorFlow).** A CNN built with the Sequential API trained from
  scratch, and a transfer-learning model built with the Functional API on top of a
  pretrained backbone (MobileNetV2 / EfficientNetB0). The input pipeline uses the
  `tf.data` API.

## Repository layout

```
notebooks/   Main Colab/Jupyter notebook (primary deliverable)
data/        Dataset is downloaded here at runtime; contents are gitignored
figures/     Plots used in the report (learning curves, confusion matrices, ROC)
results/     Experiment log (experiments.csv) and saved metrics
```

## Running the notebook

The notebook is designed to run on **Google Colab** with a free T4 GPU. It pulls
the dataset directly from Kaggle using the Kaggle API.

1. Open `notebooks/cassava_disease_classification.ipynb` in Colab.
2. Set the runtime to GPU.
3. Upload your `kaggle.json` API token when prompted (see Kaggle account settings).
4. Run cells top to bottom — the notebook handles download, preprocessing, training,
   and evaluation end-to-end.

To run locally instead, install the dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/cassava_disease_classification.ipynb
```

## Deliverables

- **Notebook:** `notebooks/cassava_disease_classification.ipynb`

## License

Code released under MIT. The Cassava Leaf Disease dataset is distributed by Kaggle
under the original competition terms — see the [competition page](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/rules)
for usage conditions.
