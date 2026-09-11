from pathlib import Path
import torch

from data import LanguageModelDataset
from model import GPT
from tokenizer import load_tokenizer

text = Path("data/shakespeare.txt").read_text(encoding="utf-8")
toke