import time
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
import os


def train_wordpiece_tokenizer(data_paths, vocab_size=64000, output_dir='tokenizer-wordpiece'):
    print("Initializing WordPiece tokenizer...")

    start_time = time.time()


    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

  
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    print(f"Starting training on {len(data_paths)} file(s)...")
    tokenizer.train(files=data_paths, trainer=trainer)
    print("Training done.")

  
    os.makedirs(output_dir, exist_ok=True)
    output_path = "/Users/simba/Downloads/monolingual-n/tokenizer-wordpiece-iit-sixtyfour-clean"
    tokenizer.save(output_path)
    print(f"Tokenizer saved to: {output_path}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training time: {elapsed:.2f} seconds")


if __name__ == "__main__":

    file_paths = [
        "/Users/simba/Downloads/monolingual-n/monolingual-clean.hi"
    ]
    train_wordpiece_tokenizer(file_paths)
