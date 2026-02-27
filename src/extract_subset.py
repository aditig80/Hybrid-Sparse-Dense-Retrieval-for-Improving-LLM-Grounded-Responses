import pandas as pd

print("Loading questions (limited columns)...")

questions = pd.read_csv(
    "data/Questions.csv",
    usecols=["Id", "Title", "Body"],
    nrows=20000,
    low_memory=False,
    encoding="latin-1"
)

print("Loading answers in chunks...")

answers_chunks = []
chunk_size = 200000

for chunk in pd.read_csv(
    "data/Answers.csv",
    usecols=["Id", "ParentId", "Score", "Body"],
    chunksize=chunk_size,
    low_memory=False,
    encoding="latin-1"
):
    filtered = chunk[chunk["ParentId"].isin(questions["Id"])]
    answers_chunks.append(filtered)

answers = pd.concat(answers_chunks)

print("Selecting highest score answer per question...")

answers = answers.sort_values("Score", ascending=False)
answers = answers.drop_duplicates(subset=["ParentId"])

print("Merging...")

merged = questions.merge(
    answers,
    left_on="Id",
    right_on="ParentId",
    suffixes=("_question", "_answer")
)

print("Saving subset...")

merged.to_csv("data/subset.csv", index=False)

print("Done. Final rows:", len(merged))