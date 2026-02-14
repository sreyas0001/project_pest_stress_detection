
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import UNet
from dataset import PestControlDataset

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).float() # BCEWithLogitsLoss expects float
        
        optimizer.zero_grad()
        
        outputs = model(images)
        # Squeeze outputs to match mask shape if needed (B, 1, H, W) -> (B, H, W) or (B, 1, H, W)
        # Model output is (B, 1, H, W). Mask is (B, H, W) from dataset usually if read(1).
        # Dataset returns mask as long tensor of shape (H, W). Wait, no.
        # Dataset returns mask (H, W). DataLoader stacks to (B, H, W).
        # We need (B, 1, H, W) for BCE.
        
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
            
        loss_bce = criterion(outputs, masks)
        loss_dice = dice_loss(outputs, masks)
        loss = loss_bce + loss_dice
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    # Global counters for metrics
    tp_total = 0
    fp_total = 0
    fn_total = 0
    intersection_total = 0
    union_total = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
                
            outputs = model(images)
            loss_bce = criterion(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = loss_bce + loss_dice
            running_loss += loss.item()
            
            # Metrics calculation
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            tp = (preds * masks).sum().item()
            fp = (preds * (1 - masks)).sum().item()
            fn = ((1 - preds) * masks).sum().item()
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item() - intersection
            
            intersection_total += intersection
            union_total += union
            
    # Calculate final metrics
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = intersection_total / (union_total + 1e-8)
            
    return running_loss / len(loader), iou, precision, recall, f1

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    # Assuming standard split logic or separate folders
    # For now, use same dir for train/val (not recommended but simple placeholder)
    # User should split data into 'train' and 'val' subfolders
    
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # If explicit train/val folders don't exist, warn user
    if not os.path.exists(train_dir):
         print(f"Train directory {train_dir} not found. Using {args.data_dir} directly (no validation split).")
         train_dataset = PestControlDataset(os.path.join(args.data_dir, 'images'), os.path.join(args.data_dir, 'masks'), patch_size=args.patch_size)
         val_loader = None
    else:
        train_dataset = PestControlDataset(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'), patch_size=args.patch_size)
        val_dataset = PestControlDataset(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'), patch_size=args.patch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    model = UNet(n_channels=5, n_classes=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss, val_iou, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), "best_model.pth")
                print("Model saved!")
        else:
             torch.save(model.state_dict(), "latest_model.pth")
             
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=256)
    
    args = parser.parse_args()
    main(args)
