import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import torch

def setup_dataset_config():
    """
    Create a YAML configuration file for the custom dataset.
    Assumes the following directory structure:
    - images/train/: Training images
    - labels/train/: Training labels (YOLO format)
    """
    # Use the absolute paths provided in required_dirs
    base_dir = 'D:\\sugarcane_training'
    
    data_yaml = {
        'path': base_dir,  # Base directory
        'train': os.path.join(base_dir, 'images', 'train'),  # Path to training images
        'val': os.path.join(base_dir, 'images', 'train'),  # Using training images for validation
        'nc': 0,  # Number of classes (will be updated)
        'names': []  # Class names (will be updated)
    }
    
    # Count the number of unique classes by checking the labels
    class_ids = set()
    labels_dir = os.path.join(base_dir, 'labels', 'train')
    if os.path.exists(labels_dir):
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
    
    data_yaml['nc'] = len(class_ids) if class_ids else 1
    data_yaml['names'] = [f'class_{i}' for i in range(data_yaml['nc'])]
    
    # Save the YAML configuration
    config_path = os.path.join(base_dir, 'dataset_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"Created dataset configuration with {data_yaml['nc']} classes")
    return config_path

def visualize_training_results(results_dir):
    """
    Generate detailed visualizations of training results
    """
    # Path to results CSV file
    results_csv = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    # Load training results
    results = pd.read_csv(results_csv)
    
    # Create output directory for graphs
    graphs_dir = os.path.join(results_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # List of metrics to plot
    metrics = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
        ('train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss'),
        ('metrics/precision', None, 'Precision'),
        ('metrics/recall', None, 'Recall'),
        ('metrics/mAP50', None, 'mAP@0.5'),
        ('metrics/mAP50-95', None, 'mAP@0.5:0.95')
    ]
    
    # Create plots for each metric
    for train_metric, val_metric, title in metrics:
        plt.figure(figsize=(10, 6))
        
        if train_metric in results.columns:
            plt.plot(results['epoch'], results[train_metric], label=f'Training {title}')
        
        if val_metric and val_metric in results.columns:
            plt.plot(results['epoch'], results[val_metric], label=f'Validation {title}')
        
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.title(f'{title} vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, f'{title.replace("@", "at").replace(":", "-").replace(" ", "_").lower()}.png'))
        plt.close()
    
    print(f"Training visualization graphs saved to {graphs_dir}")
    return graphs_dir

def print_detailed_results(results_dir):
    """
    Print detailed summary of training results
    """
    results_csv = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print("Results file not found")
        return
    
    results = pd.read_csv(results_csv)
    
    # Get the last row (final epoch results)
    final_results = results.iloc[-1]
    
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    print(f"\nTotal Training Epochs: {int(final_results['epoch'])}")
    print(f"Final Learning Rate: {final_results['lr']:.6f}")
    
    # Training losses
    print("\nFinal Loss Values:")
    print(f"  Box Loss: {final_results['train/box_loss']:.4f}")
    print(f"  Classification Loss: {final_results['train/cls_loss']:.4f}")
    print(f"  Distribution Focal Loss: {final_results['train/dfl_loss']:.4f}")
    
    # Metrics
    print("\nFinal Performance Metrics:")
    if 'metrics/precision' in final_results:
        print(f"  Precision: {final_results['metrics/precision']:.4f}")
    if 'metrics/recall' in final_results:
        print(f"  Recall: {final_results['metrics/recall']:.4f}")
    if 'metrics/mAP50' in final_results:
        print(f"  mAP@0.5: {final_results['metrics/mAP50']:.4f}")
    if 'metrics/mAP50-95' in final_results:
        print(f"  mAP@0.5:0.95: {final_results['metrics/mAP50-95']:.4f}")
    
    # Training speed
    if 'train/time' in final_results:
        print(f"\nAverage Training Time per Epoch: {final_results['train/time']:.2f} seconds")
    
    print("\nBest Performance:")
    # Find best mAP
    if 'metrics/mAP50-95' in results.columns:
        best_map = results['metrics/mAP50-95'].max()
        best_epoch = results.loc[results['metrics/mAP50-95'].idxmax(), 'epoch']
        print(f"  Best mAP@0.5:0.95: {best_map:.4f} (Epoch {int(best_epoch)})")
    
    print("\nModel Size and Speed:")
    # These values typically aren't in results.csv but can be calculated if needed
    print("  These values can be found in the model summary in the YOLO training output")
    
    print("="*50)

def train_new_model():
    """
    Train a new YOLOv8 model from scratch using a custom dataset.
    """
    # Create dataset configuration
    dataset_config = setup_dataset_config()
    
    # Initialize a new model
    model = YOLO('yolov8n.yaml')  # Start with YOLOv8 nano architecture
    
    # Automatically determine if GPU is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # If training on CPU with limited resources, you may want to reduce batch size
    batch_size = 16 if torch.cuda.is_available() else 16
    print(f"Using batch size: {batch_size}")
    
    # Train the model
    results = model.train(
        data=dataset_config,
        epochs=100,
        imgsz=640,
        batch=batch_size,
        name='new_model_training',
        verbose=True,
        patience=50,
        save=True,
        device=device  # Automatically use available device
    )
    
    # Get the results directory
    results_dir = os.path.join('runs', 'detect', 'new_model_training')
    print(f"\nTraining completed. Model saved to {results_dir}")
    
    # Generate and display detailed results
    print_detailed_results(results_dir)
    visualize_training_results(results_dir)
    
    # Validate the model on the training set to get additional metrics
    print("\nRunning validation on training set...")
    model = YOLO(os.path.join(results_dir, 'weights', 'best.pt'))
    val_results = model.val(data=dataset_config, device=device)
    
    print("\nTraining process completed successfully!")
    print(f"Final model weights: {os.path.join(results_dir, 'weights', 'best.pt')}")
    print(f"Detailed graphs available in: {os.path.join(results_dir, 'graphs')}")

if __name__ == "__main__":
    # Check if required directories exist
    required_dirs = ['D:\\sugarcane_training\\images\\train', 'D:\\sugarcane_training\\labels\\train']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} not found. Please create the required directory structure.")
            exit(1)
    
    # Install required packages if not already installed
    try:
        import ultralytics
        import matplotlib
        import pandas
    except ImportError:
        print("Installing required packages...")
        os.system('pip install ultralytics pyyaml matplotlib pandas')
    
    # Print PyTorch CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Training will use CPU. This may be slow for large datasets.")
    
    train_new_model()