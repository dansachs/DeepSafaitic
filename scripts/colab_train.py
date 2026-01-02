"""
Training script for Safaitic Siamese Network (Google Colab optimized).
Includes Google Drive mounting, training loop, checkpointing, and visualization.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Google Colab imports
try:
    from google.colab import drive
    from IPython.display import display, Image, clear_output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Not running in Google Colab - Google Drive mounting will be skipped")
    # For local testing, use basic display
    def display(*args, **kwargs):
        pass
    def clear_output(*args, **kwargs):
        pass

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from deepsafaitic import model
from deepsafaitic.model import SafaiticSiameseNet, ContrastiveLoss, euclidean_distance
from deepsafaitic.dataset import SafaiticSiameseDataset


def mount_google_drive():
    """Mount Google Drive in Colab."""
    if IN_COLAB:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
        return True
    else:
        print("Not in Colab - skipping Google Drive mount")
        return False


def create_directories():
    """Create necessary directories for checkpoints."""
    if IN_COLAB:
        checkpoint_dir = Path("/content/drive/MyDrive/safaitic_project")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    else:
        # Local directory for non-Colab runs
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    hard_negative_count = 0
    total_pairs = 0
    
    for batch in progress_bar:
        # Unpack batch: (anchor, pair, label, hard_negative)
        # hard_negative is metadata only - not used in loss computation
        if len(batch) == 4:
            anchor, pair, label, hard_negative = batch
            # Track hard negatives for statistics (detached from computation graph)
            hard_negative_count += hard_negative.sum().item()
        else:
            # Backward compatibility: old format (3 items)
            anchor, pair, label = batch
            hard_negative = None
        
        # Move to device (hard_negative stays on CPU, not needed for computation)
        anchor = anchor.to(device, non_blocking=True)
        pair = pair.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        total_pairs += label.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        embedding1 = model(anchor)
        embedding2 = model(pair)
        
        # Compute loss
        loss = criterion(embedding1, embedding2, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    hard_negative_ratio = hard_negative_count / total_pairs if total_pairs > 0 else 0.0
    
    # Clear GPU cache periodically
    if torch.cuda.is_available() and num_batches % 50 == 0:
        torch.cuda.empty_cache()
    
    return avg_loss, hard_negative_ratio


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Unpack batch: (anchor, pair, label, hard_negative)
            # hard_negative is metadata only - explicitly ignored with _
            if len(batch) == 4:
                anchor, pair, label, _ = batch  # Ignore hard_negative during validation
            else:
                # Backward compatibility: old format (3 items)
                anchor, pair, label = batch
            
            anchor = anchor.to(device, non_blocking=True)
            pair = pair.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # Forward pass
            embedding1 = model(anchor)
            embedding2 = model(pair)
            
            # Compute loss
            loss = criterion(embedding1, embedding2, label)
            
            # Compute predictions (distance < threshold = positive pair)
            distances = euclidean_distance(embedding1, embedding2)
            threshold = 1.0  # Adjust based on your margin
            predictions = (distances < threshold).float()
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            correct_predictions += (predictions == label).sum().item()
            total_predictions += label.size(0)
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, accuracy


def visualize_pairs(model, dataset, device, num_pairs=5, epoch=0):
    """
    Visualize sample pairs with their distances.
    Shows the effect of 'The Ager' (augmentations).
    """
    model.eval()
    
    # Get sample pairs
    samples = dataset.get_sample_pairs(num_pairs)
    
    fig, axes = plt.subplots(num_pairs, 3, figsize=(12, 4 * num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, sample_data in enumerate(samples):
            # Handle both old format (3 items) and new format (4 items)
            if len(sample_data) == 4:
                anchor_np, pair_np, label, hard_negative = sample_data
            else:
                anchor_np, pair_np, label = sample_data
                hard_negative = False
            # Convert to tensors
            anchor_tensor = torch.from_numpy(anchor_np).permute(2, 0, 1).unsqueeze(0).float()
            pair_tensor = torch.from_numpy(pair_np).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            anchor_tensor = (anchor_tensor - mean) / std
            pair_tensor = (pair_tensor - mean) / std
            
            anchor_tensor = anchor_tensor.to(device)
            pair_tensor = pair_tensor.to(device)
            
            # Get embeddings
            emb1 = model(anchor_tensor)
            emb2 = model(pair_tensor)
            
            # Calculate distance
            distance = euclidean_distance(emb1, emb2).item()
            
            # Display anchor
            axes[i, 0].imshow(anchor_np)
            axes[i, 0].set_title("Anchor", fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Display augmented
            axes[i, 1].imshow(pair_np)
            pair_type = "Positive" if label == 1 else "Negative"
            axes[i, 1].set_title(f"Augmented ({pair_type})", fontsize=10, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Display distance
            hard_neg_text = "\n(Hard Negative)" if hard_negative else ""
            axes[i, 2].text(0.5, 0.5, 
                           f"Distance: {distance:.3f}\n"
                           f"Label: {int(label)}\n"
                           f"Predicted: {'Same' if distance < 1.0 else 'Different'}{hard_neg_text}",
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[i, 2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}: Sample Pairs with Embedding Distances\n'
                 'Effect of "The Ager" (Aggressive Augmentations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    if IN_COLAB:
        vis_path = Path("/content/drive/MyDrive/safaitic_project") / f"visualization_epoch_{epoch}.png"
    else:
        vis_path = Path("visualizations") / f"visualization_epoch_{epoch}.png"
        vis_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {vis_path}")
    
    # Display inline in Colab
    if IN_COLAB:
        plt.show()  # Display in Colab notebook
    else:
        plt.close()


def visualize_training_samples(dataset, num_samples=10):
    """
    Quick visualization function to see raw augmented samples during training.
    Can be called from a separate Colab cell to monitor augmentation effects.
    
    Args:
        dataset: SafaiticSiameseDataset instance
        num_samples: Number of sample pairs to display
    """
    print(f"Visualizing {num_samples} sample pairs from dataset...")
    samples = dataset.get_sample_pairs(num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_data in enumerate(samples):
        if len(sample_data) == 4:
            anchor_np, pair_np, label, hard_negative = sample_data
        else:
            anchor_np, pair_np, label = sample_data
            hard_negative = False
        
        # Display anchor
        axes[i, 0].imshow(anchor_np)
        axes[i, 0].set_title("Anchor (Clean)", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Display augmented
        pair_type = "Positive" if label == 1 else "Negative"
        hard_neg_text = " (Hard Negative)" if hard_negative else ""
        axes[i, 1].imshow(pair_np)
        axes[i, 1].set_title(f"Augmented ({pair_type}){hard_neg_text}", 
                            fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.suptitle('Sample Training Pairs (Before Model Processing)\n'
                 'Showing "The Ager" Augmentation Effects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if IN_COLAB:
        plt.show()  # Display inline in Colab
    else:
        plt.savefig("training_samples_preview.png", dpi=150, bbox_inches='tight')
        print("Saved preview to: training_samples_preview.png")
        plt.close()


def main():
    """Main training function."""
    print("=" * 70)
    print("Safaitic Siamese Network Training")
    print("=" * 70)
    print()
    
    # Mount Google Drive if in Colab
    if IN_COLAB:
        mount_google_drive()
    
    # Create directories
    checkpoint_dir = create_directories()
    
    # Import model utilities for timestamped names
    try:
        sys.path.insert(0, str(project_root / "scripts"))
        from model_utils import get_timestamped_model_name
        USE_TIMESTAMPS = True
    except ImportError:
        print("Warning: model_utils not found, using non-timestamped names")
        USE_TIMESTAMPS = False
    
    # Create checkpoint path (timestamped if available)
    if USE_TIMESTAMPS:
        timestamped_name = get_timestamped_model_name("safaitic_matcher", ".pth")
        checkpoint_path = checkpoint_dir / timestamped_name
    else:
        checkpoint_path = checkpoint_dir / "safaitic_matcher.pth"
    
    # Also keep a "latest" symlink for convenience
    latest_path = checkpoint_dir / "safaitic_matcher.pth"
    
    # Verify checkpoint directory exists and is writable
    if not checkpoint_dir.exists():
        raise RuntimeError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not os.access(checkpoint_dir, os.W_OK):
        raise RuntimeError(f"Checkpoint directory is not writable: {checkpoint_dir}")
    print(f"Checkpoint directory verified: {checkpoint_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    embedding_dim = 512
    margin = 2.0
    
    # Create datasets
    print("Loading datasets...")
    full_dataset = SafaiticSiameseDataset(root_dir=str(project_root / "data" / "cleaned_glyphs"), augment=True)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset_wrapped = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create val dataset without augmentation (for proper validation)
    val_dataset_full = SafaiticSiameseDataset(root_dir=str(project_root / "data" / "cleaned_glyphs"), augment=False)
    # Use same indices as validation split
    val_indices = val_dataset_wrapped.indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Create dataloaders (optimized for Colab/GPU)
    # pin_memory=True speeds up GPU transfer, num_workers=4 for Colab (adjust based on CPU cores)
    num_workers = 4 if torch.cuda.is_available() else 2
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Creating model...")
    model = SafaiticSiameseNet(embedding_dim=embedding_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, hard_neg_ratio = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Log hard negative ratio
        if hard_neg_ratio > 0:
            print(f"  Hard negative ratio: {hard_neg_ratio:.2%}")
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create new timestamped checkpoint if available
            if USE_TIMESTAMPS:
                timestamped_name = get_timestamped_model_name("safaitic_matcher", ".pth")
                checkpoint_path = checkpoint_dir / timestamped_name
            else:
                checkpoint_path = checkpoint_dir / "safaitic_matcher.pth"
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
            
            # Add timestamp if available
            if USE_TIMESTAMPS:
                checkpoint_data['timestamp'] = datetime.now().isoformat()
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  ✓ Saved checkpoint to: {checkpoint_path}")
            
            # Update "latest" symlink/copy if using timestamps
            if USE_TIMESTAMPS:
                if latest_path.exists() or latest_path.is_symlink():
                    latest_path.unlink()
                try:
                    latest_path.symlink_to(checkpoint_path.name)
                    print(f"  ✓ Updated latest symlink: {latest_path}")
                except OSError:
                    # If symlink fails (e.g., on Windows), just copy
                    import shutil
                    shutil.copy2(checkpoint_path, latest_path)
                    print(f"  ✓ Updated latest copy: {latest_path}")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # Visualize sample pairs
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Every 5 epochs and first epoch
            print("  Generating visualization...")
            # Use the full dataset for visualization (not the subset)
            visualize_pairs(model, full_dataset, device, num_pairs=5, epoch=epoch + 1)
        
        print()
    
    # Final summary
    print("=" * 70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 70)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save training curves
    if IN_COLAB:
        curves_path = checkpoint_dir / "training_curves.png"
    else:
        curves_path = Path("training_curves.png")
    
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {curves_path}")
    
    # Display inline in Colab
    if IN_COLAB:
        plt.show()  # Display in Colab notebook
    else:
        plt.close()


if __name__ == "__main__":
    main()

