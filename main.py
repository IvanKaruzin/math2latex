import os
import json
import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import FormulaDataset, Vocab, make_collate
from model import FormulaRecognizer
from train import train_model


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default=os.environ.get("DATA_PATH", "."), help="Path to dataset folder containing annotations.csv and images")
	parser.add_argument("--tokens", default="tokens.json", help="Path to tokens.json")
	parser.add_argument("--epochs", type=int, default=15)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--model-dir", default="models")

	args = parser.parse_args()

	with open(args.tokens, "r", encoding="utf-8") as f:
		token_list = json.load(f)

	vocab = Vocab(token_list)

	dataset = FormulaDataset(args.data, vocab)

	# Split dataset: train 70%, val 20%, test 10%
	train_size = int(0.7 * len(dataset))
	val_size = int(0.2 * len(dataset))
	test_size = len(dataset) - train_size - val_size

	train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

	collate = make_collate(vocab.stoi[vocab.pad])

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate)

	model = FormulaRecognizer(vocab_size=len(vocab))

	os.makedirs(args.model_dir, exist_ok=True)

	train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		tokenizer=vocab,
		epochs=args.epochs,
		device=args.device,
	)


if __name__ == "__main__":
	main()

