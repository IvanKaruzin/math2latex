import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, test_loader, tokenizer, epochs=5, device="cuda"):
	criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi[tokenizer.pad])  # паддинг
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	model = model.to(device)

	for epoch in range(epochs):
		# ---------- TRAIN ----------
		model.train()
		total_loss = 0
		train_correct = 0
		train_total = 0
		train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

		for imgs, tokens in train_bar:
			imgs, tokens = imgs.to(device), tokens.to(device)

			optimizer.zero_grad()
			outputs = model(imgs, tokens[:, :-1])  # [B, T-1, vocab_size]

			# делаем срез по минимальной длине
			min_len = min(outputs.size(1), tokens[:, 1:].size(1))
			outputs = outputs[:, :min_len, :]
			targets = tokens[:, 1:1+min_len]

			loss = criterion(
				outputs.reshape(-1, outputs.size(-1)),
				targets.reshape(-1)
			)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()
            
			preds = outputs.argmax(-1)  # [B, T]
			pad_id = tokenizer.stoi[tokenizer.pad]
            
			for i in range(preds.size(0)):
				pred_tokens = preds[i].cpu().tolist()
				true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
                
				for j in range(min(len(pred_tokens), len(true_tokens))):
					if true_tokens[j] != pad_id:
						if pred_tokens[j] == true_tokens[j]:
							train_correct += 1
						train_total += 1
            
			acc = (train_correct / train_total) if train_total > 0 else 0.0
			train_bar.set_postfix(loss=loss.item(), acc=f"{acc:.3f}")

		avg_train_loss = total_loss / len(train_loader)
		train_accuracy = train_correct / train_total if train_total > 0 else 0

		# ---------- VALID ----------
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

				preds = outputs.argmax(-1)  # [B, T]
				pad_id = tokenizer.stoi[tokenizer.pad]
                
				for i in range(preds.size(0)):
					pred_tokens = preds[i].cpu().tolist()
					true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
					for j in range(min(len(pred_tokens), len(true_tokens))):
						if true_tokens[j] != pad_id:  # Игнорируем pad токены в истинных значениях
							if pred_tokens[j] == true_tokens[j]:
								val_correct += 1
							val_total += 1

				if len(examples) < 3:
					pred_text = tokenizer.decode([t for t in preds[0].cpu().tolist() if t != pad_id])
					true_text = tokenizer.decode([t for t in tokens[0, 1:1+min_len].cpu().tolist() if t != pad_id])
					examples.append((pred_text, true_text))

		avg_val_loss = val_loss / len(val_loader)
		val_accuracy = val_correct / val_total if val_total > 0 else 0

		print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.3f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f}")
		for i, (pred, true) in enumerate(examples):
			print(f"  EX{i+1}: pred = {pred}")
			print(f"       true = {true}")

		os.makedirs("checkpoints", exist_ok=True)
		checkpoint = { 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_val_loss,
        }

        # Save the checkpoint
		torch.save(checkpoint, os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pth"))
		print(f"Model saved: model_epoch{epoch+1}.pth")
    
	# Финальная оценка на тестовой выборке
	print("\n" + "="*50)
	print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
	print("="*50)

	model.eval()
	test_loss = 0
	test_correct = 0
	test_total = 0
	test_examples = []
	with torch.no_grad():
		test_bar = tqdm(test_loader, desc="Тестирование")
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

			preds = outputs.argmax(-1)  # [B, T]
			pad_id = tokenizer.stoi[tokenizer.pad]
            
			for i in range(preds.size(0)):
				pred_tokens = preds[i].cpu().tolist()
				true_tokens = tokens[i, 1:1+min_len].cpu().tolist()
				for j in range(min(len(pred_tokens), len(true_tokens))):
					if true_tokens[j] != pad_id:  # Игнорируем pad токены в истинных значениях
						if pred_tokens[j] == true_tokens[j]:
							test_correct += 1
						test_total += 1

			if len(test_examples) < 5:  # Показываем больше примеров для теста
				pred_text = tokenizer.decode([t for t in preds[0].cpu().tolist() if t != pad_id])
				true_text = tokenizer.decode([t for t in tokens[0, 1:1+min_len].cpu().tolist() if t != pad_id])
				test_examples.append((pred_text, true_text))

	avg_test_loss = test_loss / len(test_loader)
	test_accuracy = test_correct / test_total if test_total > 0 else 0
	print(f"\nФинальная оценка на тестовой выборке:")
	print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.3f}")
	print(f"Test Examples:")
	for i, (pred, true) in enumerate(test_examples):
		print(f"  TEST{i+1}: pred = {pred}")
		print(f"           true = {true}")
	return avg_test_loss, test_accuracy