import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss):
    checkpoint = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join("checkpoints",f"checkpoint_last.pth"))
    print(f"Checkpoint saved: epoch {epoch+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def save_best(model, optimizer, epoch, train_loss, val_loss):
    checkpoint = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, "model_best.pth")
    print(f"Checkpoint saved: epoch {epoch+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return start_epoch, best_val_loss

def train_model(model, train_loader, val_loader, test_loader, tokenizer, epochs=5, device="cuda", resume=False):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi[tokenizer.pad])
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler(device=device)

    model = model.to(device)
    best_val_loss = float('inf')
    start_epoch = 0

    if resume:
        checkpoint_path = os.path.join("checkpoints", "checkpoint_last.pth")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs} [Train]")
        
        for imgs, tokens in train_bar:
            imgs, tokens = imgs.to(device), tokens.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device=device):
                outputs = model(imgs, tokens[:, :-1])
                
                min_len = min(outputs.size(1), tokens[:, 1:].size(1))
                outputs = outputs[:, :min_len, :]
                targets = tokens[:, 1:1+min_len]
                
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            preds = outputs.argmax(-1)
            pad_id = tokenizer.stoi[tokenizer.pad]
            
            for i in range(preds.size(0)):
                pred_tokens = preds[i].cpu().tolist()
                true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
                
                for j in range(min(len(pred_tokens), len(true_tokens))):
                    if true_tokens[j] != pad_id:
                        if pred_tokens[j] == true_tokens[j]:
                            train_correct += 1
                        train_total += 1
            
            train_bar.set_postfix(loss=loss.item(), acc=f"{train_correct/train_total:.3f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        examples = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, tokens in val_bar:
                imgs, tokens = imgs.to(device), tokens.to(device)
                outputs = model(imgs, tokens[:, :-1])

                min_len = min(outputs.size(1), tokens[:, 1:].size(1))
                outputs = outputs[:, :min_len, :]
                targets = tokens[:, 1:1+min_len]

                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
                val_loss += loss.item()

                preds = outputs.argmax(-1)
                pad_id = tokenizer.stoi[tokenizer.pad]
                
                for i in range(preds.size(0)):
                    pred_tokens = preds[i].cpu().tolist()
                    true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
                    
                    for j in range(min(len(pred_tokens), len(true_tokens))):
                        if true_tokens[j] != pad_id:
                            if pred_tokens[j] == true_tokens[j]:
                                val_correct += 1
                            val_total += 1

                if len(examples) < 3:
                    pred_text = tokenizer.decode([t for t in preds[0].cpu().tolist() if t != pad_id])
                    true_text = tokenizer.decode([t for t in tokens[0, 1:1+min_len].cpu().tolist() if t != pad_id])
                    examples.append((pred_text, true_text))

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        print(f"\nEpoch {epoch+1}/{start_epoch + epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.3f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f}")
        for i, (pred, true) in enumerate(examples):
            print(f"  EX{i+1}: pred = {pred}")
            print(f"       true = {true}")
        save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_best(model, optimizer, epoch, avg_train_loss, avg_val_loss)
            print(f"Best model saved.")

        torch.cuda.empty_cache()
    
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_examples = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing...")
        for imgs, tokens in test_bar:
            imgs, tokens = imgs.to(device), tokens.to(device)
            outputs = model(imgs, tokens[:, :-1])

            min_len = min(outputs.size(1), tokens[:, 1:].size(1))
            outputs = outputs[:, :min_len, :]
            targets = tokens[:, 1:1+min_len]

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            test_loss += loss.item()

            preds = outputs.argmax(-1)
            pad_id = tokenizer.stoi[tokenizer.pad]
            
            for i in range(preds.size(0)):
                pred_tokens = preds[i].cpu().tolist()
                true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
                
                for j in range(min(len(pred_tokens), len(true_tokens))):
                    if true_tokens[j] != pad_id:
                        if pred_tokens[j] == true_tokens[j]:
                            test_correct += 1
                        test_total += 1

            if len(test_examples) < 5:
                pred_text = tokenizer.decode([t for t in preds[0].cpu().tolist() if t != pad_id])
                true_text = tokenizer.decode([t for t in tokens[0, 1:1+min_len].cpu().tolist() if t != pad_id])
                test_examples.append((pred_text, true_text))

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total if test_total > 0 else 0
    print(f"\nFinal evaluation:")
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Examples:")
    for i, (pred, true) in enumerate(test_examples):
        print(f"  TEST{i+1}: pred = {pred}")
        print(f"           true = {true}")
    

    return avg_test_loss, test_accuracy
