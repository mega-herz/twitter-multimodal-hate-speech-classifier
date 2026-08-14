from typing import Any, Dict, Optional, Union
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix



def merge_results(
    model_name: str,
    model: Any,
    y_train: Any,
    train_pred_classes: Any,
    y_val: Any,
    val_pred_classes: Any,
    class_label_map: Optional[Dict[Any, str]] = None
) -> Dict[str, Any]:
    """
    Collects evaluation metrics for train and validation predictions and 
    returns a dictionary suitable for results logging.

    Parameters:
    
    model_name : str
        Name or identifier for the trained model.
    model : Any
        Trained scikit-learn or API-compatible classifier instance.
    y_train : array-like
        Ground truth target values for training set.
    train_pred_classes : array-like
        Predicted class labels for training set.
    y_val : array-like
        Ground truth target values for validation set.
    val_pred_classes : array-like
        Predicted class labels for validation set.
    class_label_map : dict, optional
        Dictionary mapping class indices/values to display names.
    
    Returns:
    
    dict
        Dictionary containing model metadata and aggregated evaluation scores.
    """
    # Extract target names if mapping is provided
    target_names = list(class_label_map.values()) if class_label_map else None

    # Generate classification reports as dictionaries
    report_dict_train = classification_report(
        y_train,
        train_pred_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    report_dict_val = classification_report(
        y_val,
        val_pred_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Derive Weighted F1 scores
    macro_f1_train = round(report_dict_train['macro avg']['f1-score'], 4)
    macro_f1_val = round(report_dict_val['macro avg']['f1-score'], 4)
    weighted_f1_train = round(report_dict_train['weighted avg']['f1-score'], 4)
    weighted_f1_val = round(report_dict_val['weighted avg']['f1-score'], 4)
    
    # Extract hyperparameters string
    hyperparams = str(model.get_params()) if hasattr(model, 'get_params') else str(model)

    # Construct result dictionary
    result_dict = {
        'Model_name': model_name,
        'Hyperparameters': hyperparams,
        'Macro_F1_train': macro_f1_train,
        'Macro_F1_val': macro_f1_val,
        'Weighted_F1_train': weighted_f1_train,
        'Weighted_F1_val': weighted_f1_val,
        'Macro_F1_train': macro_f1_train,
        'Macro_F1_val': macro_f1_val
    }

    return result_dict





def plot_side_by_side_confusion_matrices(
    y_train_true: Union[np.ndarray, list],
    y_train_pred: Union[np.ndarray, list],
    y_val_true: Union[np.ndarray, list],
    y_val_pred: Union[np.ndarray, list],
    class_names: list,
    metric: str = "recall",
) -> plt.Figure:
  """
    Plots side-by-side normalized confusion matrices for training and
    validation sets and returns the Matplotlib Figure object for external saving or rendering.

    Args:
      y_train_true: Ground truth labels for the training set.
      y_train_pred: Predicted labels for the training set.
      y_val_true: Ground truth labels for the validation set.
      y_val_pred: Predicted labels for the validation set.
      class_names: List of string names for class labels.
      metric: Evaluation metric to normalize by ('recall' or 'precision').

    Returns:
      plt.Figure: The matplotlib figure object containing the subplots.
  """
  # Determine normalization type based on the metric string
  metric_type = "true"  # Default to recall
  if metric.lower() == "precision":
    metric_type = "pred"

  # Compute normalized confusion matrices
  cm_train = confusion_matrix(y_train_true, y_train_pred, normalize=metric_type)
  cm_val = confusion_matrix(y_val_true, y_val_pred, normalize=metric_type)

  # Create side-by-side subplots and capture the figure object
  fig, axes = plt.subplots(1, 2, figsize=(16, 7))
  display_metric = metric.capitalize()

  # 1. Training Confusion Matrix
  sns.heatmap(
      cm_train,
      annot=True,
      fmt=".2f",
      cmap="Blues",
      vmin=0.0,
      vmax=1.0,
      xticklabels=class_names,
      yticklabels=class_names,
      ax=axes[0],
      cbar=False,
  )
  axes[0].set_title(
      f"Training Set ({display_metric})", fontsize=12, fontweight="bold"
  )
  axes[0].set_xlabel("Predicted Label", fontsize=10)
  axes[0].set_ylabel("True Label", fontsize=10)
  axes[0].tick_params(axis="x", rotation=45)
  axes[0].tick_params(axis="y", rotation=0)

  # 2. Validation Confusion Matrix
  sns.heatmap(
      cm_val,
      annot=True,
      fmt=".2f",
      cmap="Blues",
      vmin=0.0,
      vmax=1.0,
      xticklabels=class_names,
      yticklabels=class_names,
      ax=axes[1],
      cbar=True,
  )
  axes[1].set_title(
      f"Validation Set ({display_metric})", fontsize=12, fontweight="bold"
  )
  axes[1].set_xlabel("Predicted Label", fontsize=10)
  axes[1].set_ylabel("True Label", fontsize=10)
  axes[1].tick_params(axis="x", rotation=45)
  axes[1].tick_params(axis="y", rotation=0)

  plt.tight_layout()

  # Return the figure object instead of calling plt.show() inside
  return fig