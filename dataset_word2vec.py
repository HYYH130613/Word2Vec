from datasets import load_dataset
from _token import token


ds = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1", 
    streaming=True,
    token = token
    )

ds = ds["train"].filter(lambda x: len(x["text"]) > 200)

for x in ds:
    print(x["text"][:200])
    break