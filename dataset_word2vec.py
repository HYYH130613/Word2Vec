from datasets import load_dataset
from _token import token


ds = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1", 
    streaming=True,
    token = token
    )


for example in ds["train"]:
    print(example["text"])
    break