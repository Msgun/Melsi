import csv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

"""
svg_names.csv content:
-10__round_label.svg
-10__round_label_mod.svg
0-50.svg
0-accident.svg
0-panne.svg
...
"""

filenames = []
with open("svg_names.csv", "r") as f:
    for row in csv.reader(f):
        if row and row[0].strip().lower().endswith(".svg"):
            filenames.append(row[0].strip())

# remove .svg for embedding text
texts = [name.replace(".svg", "") for name in filenames]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

with open("svg_embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "filenames": filenames,
            "embeddings": embeddings
        },
        f
    )

print("saved", len(texts), "embeddings")
