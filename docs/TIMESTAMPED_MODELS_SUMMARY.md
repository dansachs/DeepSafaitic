# Timestamped Model Checkpoints - Summary

## What Changed

Models are now saved with timestamps in their filenames, making it easy to track different training runs.

## New File Format

**Before:**
- `safaitic_matcher.pth` (overwrites previous)

**After:**
- `safaitic_matcher_2024-12-30_14-30-45.pth` (unique per training run)
- `safaitic_matcher.pth` (symlink/copy to latest, for backward compatibility)

## Files Created/Updated

### New Files:
1. **`model_utils.py`** - Utility functions for timestamped models
   - `get_timestamped_model_name()` - Generate timestamped filename
   - `find_latest_model()` - Find most recent model in directory
   - `list_models()` - List all models sorted by date

2. **`list_models.py`** - Script to list all available models
   ```bash
   python list_models.py models/
   ```

3. **`NOTEBOOK_UPDATE_GUIDE.md`** - Guide for updating the notebook

### Updated Files:
1. **`colab_train.py`** - Training script now saves timestamped models
2. **`scan_and_visualize.py`** - Scanner automatically finds latest model

## How It Works

### During Training

When validation loss improves:
1. Creates timestamped checkpoint: `safaitic_matcher_2024-12-30_14-30-45.pth`
2. Saves model with timestamp in metadata
3. Updates `safaitic_matcher.pth` symlink/copy to point to latest

### Using Models

**Option 1: Use Latest Automatically**
```bash
# Scanner finds latest automatically
python scan_and_visualize.py stone.jpg \
    --checkpoint models/ \
    --glyphs cleaned_glyphs
```

**Option 2: Specify Exact Model**
```bash
# Use specific timestamped model
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher_2024-12-30_14-30-45.pth \
    --glyphs cleaned_glyphs
```

**Option 3: Use "Latest" Symlink**
```bash
# Still works for backward compatibility
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

## Benefits

1. **Track Training History** - See when each model was trained
2. **Compare Models** - Keep multiple versions for comparison
3. **Rollback** - Easily use older models if needed
4. **Backward Compatible** - `safaitic_matcher.pth` still works

## Example Output

```
Training...
Epoch 1/50
  ✓ Saved checkpoint to: safaitic_matcher_2024-12-30_14-30-45.pth
  ✓ Updated latest symlink: safaitic_matcher.pth

Epoch 5/50
  ✓ Saved checkpoint to: safaitic_matcher_2024-12-30_14-35-12.pth
  ✓ Updated latest symlink: safaitic_matcher.pth
```

## Listing Models

```bash
$ python list_models.py models/

============================================================
Available Model Checkpoints
============================================================
Directory: models

Found 3 model(s):

1. safaitic_matcher_2024-12-30_14-35-12.pth
   Timestamp: 2024-12-30 14:35:12
   Size: 134.23 MB
   Path: models/safaitic_matcher_2024-12-30_14-35-12.pth

2. safaitic_matcher_2024-12-30_14-30-45.pth
   Timestamp: 2024-12-30 14:30:45
   Size: 134.23 MB
   Path: models/safaitic_matcher_2024-12-30_14-30-45.pth

3. safaitic_matcher_2024-12-29_10-15-30.pth
   Timestamp: 2024-12-29 10:15:30
   Size: 134.23 MB
   Path: models/safaitic_matcher_2024-12-29_10-15-30.pth

============================================================
Latest model: safaitic_matcher_2024-12-30_14-35-12.pth
Use this path: models/safaitic_matcher_2024-12-30_14-35-12.pth
============================================================
```

## Updating Your Notebook

If you're using the Jupyter notebook, see `NOTEBOOK_UPDATE_GUIDE.md` for step-by-step instructions.

The changes are already in `colab_train.py`, so if you're using that script, you're all set!

## Backward Compatibility

- Old code using `safaitic_matcher.pth` still works (points to latest)
- Scanner automatically finds latest if given a directory
- All existing functionality preserved

## Next Steps

1. **Train a new model** - It will automatically get a timestamp
2. **List models** - Use `python list_models.py` to see all versions
3. **Compare models** - Test different versions on the same images
4. **Keep good models** - Don't delete timestamped files, only the symlink updates

Enjoy tracking your model versions! 🎉

