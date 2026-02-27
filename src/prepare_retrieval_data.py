import pandas as pd
import re
import pickle

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

print("Loading subset...")

df = pd.read_csv("data/subset.csv", encoding="latin-1")

print("Cleaning text...")

df["query"] = df["Title"] + " " + df["Body_question"]
df["query"] = df["query"].apply(clean_text)

df["document"] = df["Body_answer"].apply(clean_text)

print("Dropping empty rows...")

df = df.dropna(subset=["query", "document"])

# Keep only necessary columns
documents = df[["Id_answer", "document"]].reset_index(drop=True)
queries = df[["Id_answer", "query"]].reset_index(drop=True)

print("Saving processed files...")

documents.to_csv("data/documents.csv", index=False)
queries.to_csv("data/queries.csv", index=False)

# Create ID to index mapping
id_to_index = {
    doc_id: idx for idx, doc_id in enumerate(documents["Id_answer"])
}

with open("data/id_mapping.pkl", "wb") as f:
    pickle.dump(id_to_index, f)

print("Done.")
print("Total samples:", len(documents))