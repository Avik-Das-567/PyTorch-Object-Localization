# Object Localization with PyTorch and EfficientNet-B0

A deep learning project for **single-object localization**: given an image, the model predicts a tight bounding box around the target object using supervised regression on bounding-box coordinates.

This implementation follows the object-localization workflow from the notebook and is centered on:
- a custom PyTorch dataset for image + bounding-box pairs,
- Albumentations-based geometric augmentation with bounding-box awareness,
- a pretrained `timm` backbone,
- a training loop that optimizes bounding-box regression with mean squared error,
- validation-based checkpointing and inference visualizations.

## Problem Statement

Object localization asks the model to identify **where** the object is in an image rather than classify **what** the image contains. Instead of returning a class label, the network regresses four coordinates describing a bounding box:
- `xmin`
- `ymin`
- `xmax`
- `ymax`

The training data in this project consists of images paired with bounding-box annotations in **Pascal VOC** format. The dataset contains object categories such as **eggplant**, **cucumber**, and **mushroom**, with image dimensions stored alongside the coordinates.

## Project Overview

The pipeline begins by reading the CSV annotation file and visualizing raw image/bounding-box pairs to verify the annotation format. The data is then split into training and validation subsets using an 80/20 split. Next, the images are resized to a fixed resolution and augmented with random flips and rotation while preserving bounding-box consistency.

A custom `torch.utils.data.Dataset` implementation loads each image, applies synchronized augmentations to both the image and its bounding box, converts the result to a tensor, and returns the pair to the dataloader. The model uses a pretrained EfficientNet-B0 backbone from `timm` and is configured to output four values corresponding to the bounding-box coordinates. Training minimizes MSE between predicted and ground-truth boxes, and the best checkpoint is saved whenever validation loss improves.

For inference, the saved checkpoint is restored and predictions are compared against ground truth using visualization utilities.

## Learning Objectives

- Understand the localization dataset and annotation format
- Build a custom dataset for image and bounding-box regression
- Apply Albumentations for spatial augmentation with bbox support
- Load a pretrained convolutional neural network using `timm`
- Implement training and evaluation loops for localization
- Predict and visualize bounding boxes on unseen images

## Dataset

The notebook uses a dataset cloned from a GitHub repository and read from a CSV file containing:
- `img_path`
- `xmin`, `ymin`, `xmax`, `ymax`
- `width`, `height`
- `label`

The notebook preview shows **186 annotated samples**. After the train/validation split, the dataset contains:
- **148 training examples**
- **38 validation examples**

Each annotation represents one object instance per image. The bounding boxes are stored in absolute pixel coordinates.

## Data Inspection

Before training, the notebook visualizes sample annotations directly on top of the original RGB images using OpenCV and Matplotlib. This step confirms that:
- the CSV coordinates align with the image content,
- the bounding box format is correct,
- the target object occupies a meaningful portion of the image,
- the localization task is suitable for bounding-box regression.

This inspection is repeated on both original and augmented samples to verify that transformations preserve label consistency.

## Augmentation Strategy

Albumentations is used with `bbox_params` so that image transforms and bounding boxes remain synchronized.

### Training augmentations
- Resize to `140 × 140`
- Random horizontal flip (`p=0.5`)
- Random vertical flip (`p=0.5`)
- Random rotation

### Validation augmentations
- Resize to `140 × 140`

Bounding boxes are declared in `pascal_voc` format and the augmentation pipeline is configured with `label_fields=['class_labels']` so that the library can transform the coordinates correctly.

## Custom Dataset Design

The `ObjLocDataset` class extends `torch.utils.data.Dataset` and encapsulates the full sample-loading logic.

### Responsibilities
1. Read a row from the annotation dataframe.
2. Extract the four bounding-box coordinates.
3. Load the corresponding image from disk.
4. Convert the image from BGR to RGB.
5. Apply the augmentation pipeline, if provided.
6. Convert the image to a float tensor with channel-first layout.
7. Convert the bounding box to a tensor.

### Returned sample
Each dataset item returns:
- an image tensor of shape `[3, 140, 140]`
- a bounding-box tensor of shape `[4]`

The implementation keeps the dataset focused and reusable across both training and validation pipelines.

## Data Loading

The dataset is wrapped by standard PyTorch dataloaders.

Configuration used in the notebook:
- batch size: `16`
- training loader: shuffled
- validation loader: sequential

The first batch in the notebook has shape:
- images: `[16, 3, 140, 140]`
- bounding boxes: `[16, 4]`

This matches the expected input/output contract for the regression model.

## Model Architecture

The model is built with `timm.create_model` using **EfficientNet-B0** as a pretrained backbone.

### Key design choices
- `pretrained=True` to leverage ImageNet features
- `num_classes=4` so the model outputs four regression values
- a minimal forward pass that returns:
  - predictions only during inference
  - predictions plus loss during training when ground-truth boxes are available

### Regression objective
The model uses **Mean Squared Error (MSE)** between the predicted and target bounding-box coordinates. This makes the task a direct regression problem rather than a detection pipeline with anchors, objectness scores, or class logits.

## Training and Evaluation

Two dedicated functions handle the optimization workflow.

### `train_fn`
- switches the model to training mode
- iterates over training batches
- moves images and targets to the selected device
- computes predictions and MSE loss
- performs backpropagation
- updates the optimizer
- returns the average training loss

### `eval_fn`
- switches the model to evaluation mode
- disables gradient computation
- computes validation loss across the validation dataloader
- returns the average validation loss

### Optimizer
The notebook uses **Adam** with learning rate `0.001`.

### Checkpointing
The best-performing model is saved to `best_model.pt` whenever validation loss improves. This keeps the strongest checkpoint from the full training run.

## Training Configuration

The notebook uses the following core hyperparameters:

- image size: `140`
- batch size: `16`
- learning rate: `0.001`
- epochs: `40`
- backbone: `efficientnet_b0`
- device: `cuda`
- output dimensions: `4`

This setup balances a lightweight input resolution with a strong pretrained backbone for coordinate regression.

## Project Workflow

1. **Load annotations** from the CSV file and inspect the bounding-box schema.
2. **Visualize sample images** with boxes overlaid to validate the data.
3. **Split the dataset** into training and validation subsets.
4. **Define augmentations** that resize images and transform boxes consistently.
5. **Implement a custom dataset** that returns image tensors and bounding-box tensors.
6. **Create dataloaders** for batched training and evaluation.
7. **Build a pretrained regression model** on top of EfficientNet-B0.
8. **Define training and validation loops** with MSE loss.
9. **Train for multiple epochs** while saving the best checkpoint by validation loss.
10. **Restore the best model** and run inference on validation images.
11. **Compare predicted boxes** against ground truth visually.

## Inference

For inference, the notebook:
- loads the saved checkpoint from `best_model.pt`,
- switches the model to evaluation mode,
- selects a validation image,
- runs a forward pass to predict the bounding box,
- visualizes predicted and ground-truth boxes for comparison.

The notebook repeats this inference step on multiple validation samples to show how the model generalizes across examples.

## Training Outcome

During the recorded training run, validation loss steadily decreased early in training and the best checkpoint was achieved at **epoch 38** with a validation loss of approximately **33.71**. The final epoch still remained in the same loss range, which indicates that the model had learned a stable regression mapping for bounding-box prediction.

## Technologies Used

- Python
- PyTorch
- timm
- Albumentations
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## Notes

- Bounding boxes are handled in absolute pixel coordinates and converted through Albumentations using Pascal VOC semantics.
- The dataset pipeline is designed for a single bounding box per image.
- The forward pass is intentionally simple to keep the focus on localization rather than full object detection.
- Visual verification is an important part of the workflow because bounding-box regression is sensitive to incorrect coordinate handling.
