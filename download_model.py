from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"

print("Downloading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./models")
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir="./models")
print("Model downloaded successfully!")