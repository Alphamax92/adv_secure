"""
PubMedBERT Adversarial Detection Training Script
Fine-tunes PubMedBERT on the telemedicine adversarial detection dataset.
Includes:
- Temperature scaling for probability calibration.
- Precision-recall curve analysis for optimal threshold tuning.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    PrecisionRecallDisplay,
    brier_score_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
CSV_PATH = "/content/full_telemedicine_dataset.csv"
OUTPUT_DIR = "/content/pubmedbert_telemedicine_model"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 32
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Calibration and Threshold Tuning Parameters
TARGET_PRECISION = 0.95 # Target precision for the positive class (adversarial)
CALIBRATION_PARAMS_FILE = os.path.join(OUTPUT_DIR, "calibration_params.pt")

# --- Utility Functions ---

def setup_device():
    """Setup and return the appropriate device (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_name(0)}: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def load_dataset(csv_path):
    """Load and validate the dataset"""
    print("Loading dataset...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    if df.isnull().sum().any():
        print("Warning: Missing values found, dropping rows...")
        df = df.dropna()
    text_lengths = df['prompt'].str.split().str.len()
    print(f"Text length - Mean: {text_lengths.mean():.2f}, Min: {text_lengths.min()}, Max: {text_lengths.max()}")
    return df

def prepare_data(df, test_size=0.1, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    print("Splitting dataset...")
    X = df['prompt'].tolist()
    y = df['label'].tolist()
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def tokenize_data(texts, labels, tokenizer, max_length=32):
    """Tokenize texts and create dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    dataset = Dataset.from_dict({'text': texts, 'labels': labels})
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Calibration and Threshold Tuning ---

class TemperatureScaler(nn.Module):
    """
    A module for temperature scaling.
    Scales logits by a single learned parameter T.
    """
    def __init__(self, initial_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits):
        return logits / self.temperature

def calibrate_model(model, val_dataset, device):
    """
    Calibrates the model's probabilities using temperature scaling on the validation set.
    Returns the optimal temperature.
    """
    print("\n--- Calibrating Model Probabilities (Temperature Scaling) ---")
    model.eval()

    # Get raw logits and true labels from validation set
    val_logits = []
    val_labels = []

    # Create a DataLoader for the validation set
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_logits.append(outputs.logits.cpu())
            val_labels.append(labels.cpu())

    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)

    # Initialize temperature scaler
    scaler = TemperatureScaler().to(device)

    # Define optimizer and loss function
    # Use BCEWithLogitsLoss for binary classification, as it's numerically stable
    # and works directly with logits.
    criterion = nn.BCEWithLogitsLoss().to(device)

    # LBFGS is often recommended for temperature scaling as it can converge faster
    # for this specific optimization problem.
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        # Select logits for the positive class (index 1) to match target shape
        loss = criterion(scaler(val_logits.to(device))[:, 1], val_labels.float().to(device))
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temperature = scaler.temperature.item()
    print(f"Optimal Temperature: {optimal_temperature:.4f}")

    # Plot calibration curve
    calibrated_probs = torch.sigmoid(torch.tensor(val_logits / optimal_temperature)).numpy()
    true_labels_np = val_labels.numpy()

    prob_true, prob_pred = calibration_curve(true_labels_np, calibrated_probs[:, 1], n_bins=10) # For positive class

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title('Calibration Curve (Temperature Scaling)')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate Brier Score (lower is better)
    brier = brier_score_loss(true_labels_np, calibrated_probs[:, 1])
    print(f"Brier Score (Calibrated): {brier:.4f}")

    return optimal_temperature

def find_optimal_threshold(y_true, y_scores, target_precision=0.95):
    """
    Computes the precision-recall curve and finds a threshold that meets
    the target precision while maximizing recall.

    Args:
        y_true (np.array): True binary labels.
        y_scores (np.array): Predicted probabilities for the positive class.
        target_precision (float): The minimum precision to achieve.

    Returns:
        float: The optimal threshold.
    """
    print(f"\n--- Finding Optimal Threshold (Target Precision >= {target_precision:.2f}) ---")
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Find the threshold that meets the target precision
    optimal_threshold = 0.5 # Default if no threshold meets the target
    max_recall_at_target_precision = -1

    for p, r, t in zip(precision, recall, thresholds):
        if p >= target_precision:
            if r > max_recall_at_target_precision:
                max_recall_at_target_precision = r
                optimal_threshold = t

    print(f"Optimal Threshold found: {optimal_threshold:.4f} (achieving Precision: {max_recall_at_target_precision:.4f} at this threshold)")

    # Plot Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name="PubMedBERT")
    display.plot(ax=ax)
    ax.axhline(y=target_precision, color='r', linestyle='--', label=f'Target Precision ({target_precision:.2f})')
    ax.axvline(x=max_recall_at_target_precision, color='g', linestyle=':', label=f'Recall at Target Precision ({max_recall_at_target_precision:.2f})')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return optimal_threshold

# --- Main Training and Evaluation Pipeline ---

def train_model_pipeline(model, tokenizer, train_data, val_data, output_dir, epochs, batch_size, learning_rate):
    """
    Trains the PubMedBERT model and returns the trainer, along with
    validation logits and true labels for calibration.
    """
    print("Preparing datasets for training...")
    train_dataset = tokenize_data(train_data[0], train_data[1], tokenizer)
    val_dataset = tokenize_data(val_data[0], val_data[1], tokenizer)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Get raw logits and true labels for validation set for calibration
    val_predictions_output = trainer.predict(val_dataset)
    val_logits = val_predictions_output.predictions
    val_true_labels = val_predictions_output.label_ids

    return trainer, val_logits, val_true_labels

def evaluate_model_with_calibration(trainer, test_data, tokenizer, output_dir, optimal_temperature, optimal_threshold):
    """
    Evaluate the trained model using calibrated probabilities and the optimal threshold.
    """
    print("\n--- Evaluating Model on Test Set with Calibration ---")
    test_dataset = tokenize_data(test_data[0], test_data[1], tokenizer)

    # Get raw logits from the model on the test set
    test_predictions_output = trainer.predict(test_dataset)
    test_logits = test_predictions_output.predictions
    y_true = test_predictions_output.label_ids

    # Apply temperature scaling to test logits
    calibrated_test_logits = test_logits / optimal_temperature
    calibrated_test_probs = torch.sigmoid(torch.tensor(calibrated_test_logits)).numpy()

    # Use the optimal threshold for binary classification
    y_pred = (calibrated_test_probs[:, 1] >= optimal_threshold).astype(int) # Class 1 is adversarial

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Test Results (Calibrated & Thresholded):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Adversarial'])
    print("\nDetailed Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Adversarial'],
               yticklabels=['Normal', 'Adversarial'])
    plt.title('PubMedBERT Adversarial Detection - Confusion Matrix (Calibrated)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_calibrated.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'optimal_temperature': optimal_temperature,
        'optimal_threshold': optimal_threshold
    }

    with open(os.path.join(OUTPUT_DIR, 'evaluation_results_calibrated.txt'), 'w') as f:
        f.write("PubMedBERT Adversarial Detection - Evaluation Results (Calibrated)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation completed: {datetime.now()}\n\n")
        f.write(f"Optimal Temperature: {optimal_temperature:.4f}\n")
        f.write(f"Optimal Threshold (for Precision >= {TARGET_PRECISION:.2f}): {optimal_threshold:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"True Negatives: {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n")
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives: {cm[1,1]}\n")

    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'evaluation_results_calibrated.txt')}")
    return results

def main():
    print("=" * 60)
    print("PubMedBERT Adversarial Detection Training with Calibration & Threshold Tuning")
    print("=" * 60)

    device = setup_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = load_dataset(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Please upload {CSV_PATH} to Colab")
        return

    train_data, val_data, test_data = prepare_data(df)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    print(f"Model loaded and moved to {device}")

    # Train model and get validation logits/labels
    trainer, val_logits, val_true_labels = train_model_pipeline(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    # --- Calibration ---
    optimal_temperature = calibrate_model(model, tokenize_data(val_data[0], val_data[1], tokenizer), device)

    # --- Threshold Tuning ---
    # Apply temperature scaling to validation logits to get calibrated probabilities
    calibrated_val_probs = torch.sigmoid(torch.tensor(val_logits / optimal_temperature)).numpy()

    # We need probabilities for the positive class (label 1)
    optimal_threshold = find_optimal_threshold(val_true_labels, calibrated_val_probs[:, 1], TARGET_PRECISION)

    # Save calibration parameters and optimal threshold
    torch.save({
        'temperature': optimal_temperature,
        'threshold': optimal_threshold
    }, CALIBRATION_PARAMS_FILE)
    print(f"Calibration parameters and optimal threshold saved to {CALIBRATION_PARAMS_FILE}")

    # --- Final Evaluation on Test Set with Calibrated Model ---
    results = evaluate_model_with_calibration(
        trainer=trainer,
        test_data=test_data,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        optimal_temperature=optimal_temperature,
        optimal_threshold=optimal_threshold
    )

    print("\n" + "=" * 60)
    print("Training, Calibration, and Evaluation Completed!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Final Test F1-Score (Calibrated): {results['f1']:.4f}")
    print("=" * 60)

    return trainer, results

if __name__ == "__main__":
    main()