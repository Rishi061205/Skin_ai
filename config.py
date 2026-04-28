

import torch

# Paths
TRAIN_DIR       = "data/train"           
VAL_DIR         = "data/val"             
TEST_DIR        = "data/test"            
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# Image settings
IMG_SIZE = 224                           # EfficientNet-B3 works well at 224x224 (smaller than 300x300 saves memory)
MEAN     = [0.485, 0.456, 0.406]        # ImageNet mean (pretrained model expects this)
STD      = [0.229, 0.224, 0.225]        # ImageNet std

# Training hyperparameters 
BATCH_SIZE    = 16       
NUM_EPOCHS    = 30
LEARNING_RATE = 3e-4     
BACKBONE_LR   = 3e-5     
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7       


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
