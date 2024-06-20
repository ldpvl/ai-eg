import spacy
from spacy.training.example import Example
import random

# Load a small model
nlp = spacy.load("en_core_web_sm")

# Add a text classifier to the pipeline if not already present
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Add labels to the text classifier
categories = ["POSITIVE", "NEGATIVE"]
for category in categories:
    textcat.add_label(category)

# Example training data
train_data = [
    ("I love this product!", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("This is the worst service ever.", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]

# Convert training data to spaCy's Example format
examples = []
for text, annotation in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotation)
    examples.append(example)

# Train the model
optimizer = nlp.initialize()
for epoch in range(10):
    random.shuffle(examples)
    losses = {}
    nlp.update(examples, drop=0.5, losses=losses)
    print(f"Epoch {epoch} - Losses: {losses}")

# Save the fine-tuned model
nlp.to_disk("fine_tuned_small_model")
