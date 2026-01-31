# Google Colab Training Guide

## Quick Start for Google Colab

The training script now **automatically detects** if you're running in Google Colab and will:
1. ✅ Automatically mount Google Drive
2. ✅ Save all models to Google Drive instead of Colab's temporary storage
3. ✅ Preserve your models even after the Colab session ends

## Step 1: Upload Your Project to Google Colab

1. **Option A - Upload as ZIP:**
   - Compress your entire project folder
   - Upload to Google Colab and extract it

2. **Option B - Clone from GitHub:**
   ```python
   !git clone https://github.com/your-repo/your-project.git
   %cd your-project
   ```

## Step 2: Install Dependencies

```python
!pip install torch numpy pyyaml
# Add any other dependencies your project needs
```

## Step 3: Run Training

**That's it!** Just run your training script normally:

```python
!python train_multi_dc.py
```

The script will automatically:
- Detect that it's running in Colab
- Mount Google Drive (you'll need to authorize it)
- Save models to: `/content/drive/MyDrive/thesis_models/results/full_training/`

## Step 4: Resume Training from Saved Models

To resume training from a previously saved model:

```python
# Set the resume path in the script
RESUME_MODEL_DIR = "/content/drive/MyDrive/thesis_models/results/full_training/run_seed_1/models"
```

Or edit the script directly at line ~131 in `train_multi_dc.py`.

## Customizing Google Drive Path

By default, models are saved to:
```
/content/drive/MyDrive/thesis_models/
```

To change this, edit line ~45 in `train_multi_dc.py`:

```python
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/your_custom_folder"
```

## Where Are My Models Saved?

**In Google Colab:**
- Path: `/content/drive/MyDrive/thesis_models/results/experiment_name/run_seed_1/models/`
- You can access these files through Google Drive at: `MyDrive/thesis_models/`

**Running Locally (on your PC):**
- Path: `D:\thuan\thesis\...\results\experiment_name\run_seed_1\models\`
- The script automatically uses local storage when not in Colab

## Force Google Drive Usage (Optional)

If you want to force Google Drive usage even when not in Colab, set this in `train_multi_dc.py` (line ~42):

```python
USE_GOOGLE_DRIVE = True
```

## Checking GPU Status in Colab

Make sure you're using GPU in Colab:

1. Go to **Runtime → Change runtime type**
2. Select **T4 GPU** or **A100 GPU** (if available)
3. Save

Verify GPU is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Tips for Long Training Sessions

1. **Keep Colab Active:**
   - Colab disconnects after ~12 hours or if idle
   - Use browser extensions to keep the tab active

2. **Monitor Progress:**
   - Models are saved incrementally when performance improves
   - Check `/content/drive/MyDrive/thesis_models/` periodically

3. **Resume After Disconnection:**
   - Your models are safe in Google Drive
   - Just set `RESUME_MODEL_DIR` and restart the script

## Troubleshooting

**Problem:** "Google Drive not mounted"
- **Solution:** Manually mount it first:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

**Problem:** "Permission denied" when saving
- **Solution:** Check that you have write access to Google Drive and the folder exists

**Problem:** Script saves locally instead of Google Drive
- **Solution:** Check that you authorized Google Drive access when prompted

---

## Summary

✅ **Automatic detection** - No code changes needed between local and Colab  
✅ **Persistent storage** - Models saved to Google Drive  
✅ **Resume capability** - Continue training across sessions  
✅ **Simplified paths** - Now using `results/experiment_name` instead of complex nested folders
