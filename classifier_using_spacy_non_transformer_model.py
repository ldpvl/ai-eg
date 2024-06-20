import spacy
from spacy.training.example import Example
import random

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Add the text classifier to the pipeline if it's not already there
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Add labels to the text classifier
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Example training data
train_data = [
    ("I love this product!", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("This is the worst service ever.", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]

# Convert training data to spaCy's Example format
examples = []
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Create an optimizer for just the textcat component
optimizer = nlp.create_optimizer()

# Get the components you want to train
train_components = ["textcat"]

# Disable other components for training
disabled_pipes = [pipe for pipe in nlp.pipe_names if pipe not in train_components]

with nlp.disable_pipes(*disabled_pipes):
    # Train the textcat component
    for epoch in range(10):
        random.shuffle(examples)
        losses = {}
        nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)
        print(f"Epoch {epoch} - Losses: {losses}")

# Save the fine-tuned model
output_dir = "fine_tuned_textcat_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Test the model
test_text = "The service was fantastic!"
doc = nlp(test_text)
print(test_text, doc.cats)
