# Updating Notebook for Timestamped Models

## Changes Needed

The notebook (`safaitic_training.ipynb`) needs to be updated to use timestamped model names. Here are the changes:

## Cell Updates

### 1. Add Import Cell (after model imports)

Add a new cell after the model imports:

```python
# Model utilities for timestamped checkpoints
from model_utils import get_timestamped_model_name
from datetime import datetime
```

### 2. Update Checkpoint Path Cell

**Find this cell:**
```python
checkpoint_dir = Path('/content/drive/MyDrive/safaitic_project')
checkpoint_path = checkpoint_dir / "safaitic_matcher.pth"
```

**Replace with:**
```python
checkpoint_dir = Path('/content/drive/MyDrive/safaitic_project')

# Create timestamped checkpoint path
from model_utils import get_timestamped_model_name
timestamped_name = get_timestamped_model_name("safaitic_matcher", ".pth")
checkpoint_path = checkpoint_dir / timestamped_name

# Also keep a "latest" symlink for convenience
latest_path = checkpoint_dir / "safaitic_matcher.pth"
```

### 3. Update Best Checkpoint Saving

**Find this code:**
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    # Save checkpoint if validation loss improved
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"  ✓ Saved checkpoint to: {checkpoint_path}")
```

**Replace with:**
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    
    # Create new timestamped checkpoint
    timestamped_name = get_timestamped_model_name("safaitic_matcher", ".pth")
    checkpoint_path = checkpoint_dir / timestamped_name
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'timestamp': datetime.now().isoformat(),
    }, checkpoint_path)
    print(f"  ✓ Saved checkpoint to: {checkpoint_path}")
    
    # Update "latest" symlink (remove old if exists, create new)
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
```

### 4. Update Periodic Checkpointing (Optional)

**Find this code:**
```python
if (epoch + 1) % 10 == 0:
    periodic_path = checkpoint_dir / f"safaitic_matcher_epoch_{epoch+1}.pth"
    torch.save({...}, periodic_path)
```

**Replace with:**
```python
if (epoch + 1) % 10 == 0:
    # Use timestamped name for periodic checkpoints too
    periodic_name = get_timestamped_model_name(f"safaitic_matcher_epoch_{epoch+1}", ".pth")
    periodic_path = checkpoint_dir / periodic_name
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'timestamp': datetime.now().isoformat(),
    }, periodic_path)
    print(f"  ✓ Saved periodic checkpoint to: {periodic_path}")
```

## Quick Update Script

Alternatively, you can use the updated `colab_train.py` script which already has these changes. The notebook cells can call functions from that script, or you can copy the logic.

## Verification

After updating, when you train a model, you should see:
- Checkpoints saved as: `safaitic_matcher_2024-12-30_14-30-45.pth`
- A symlink/copy at: `safaitic_matcher.pth` (points to latest)

## Using Timestamped Models

### In Scanner

The scanner automatically finds the latest model if you point it to a directory:

```python
# Automatically finds latest
python scan_and_visualize.py stone.jpg \
    --checkpoint models/ \
    --glyphs cleaned_glyphs

# Or specify exact model
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher_2024-12-30_14-30-45.pth \
    --glyphs cleaned_glyphs
```

### List Available Models

```bash
python list_models.py models/
```

This will show all available models sorted by date.

