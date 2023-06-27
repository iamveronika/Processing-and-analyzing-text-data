import random
import spacy
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
from spacy.training import Example
import json

nlp = spacy.load("en_core_web_md")

train_data = [
    ("We're going to sent transferring funds.", {"entities": [(20, 38, "MONEY")]}),
    ("Today I`m receiving money transactions.", {"entities": [(20, 25, "MONEY")]}),
    ("This current balance is very important for me.", {"entities": [(13, 20, "MONEY")]}),
    ("This bank is popular for seving money among milioners.", {"entities": [(5, 9, "MONEY")]})
]

ner = nlp.get_pipe("ner")
ner.add_label("MONEY")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()

    for _ in range(100):
        random.shuffle(train_data)
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer)

nlp.to_disk("custom_ner_model")

text = "I love to transferring money to increase my bank balance. How I can do it?"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)


with open('banks.json') as file:
    data = json.load(file)

print("")

config = {
    "threshold": 0.5,
    "model": DEFAULT_MULTI_TEXTCAT_MODEL
}
textcat = nlp.add_pipe("textcat_multilabel", config=config)

textcat.add_label("savings")

train_examples = []
for dialogue in data:
    for turn in dialogue["turns"]:
        if turn["speaker"] == "USER":
            text = turn["utterance"]
            intent = turn["frames"][0]["state"]["active_intent"]
            label = {"cats": {"TransferMoney": intent == "TransferMoney"}}
            example = Example.from_dict(nlp.make_doc(text), label)
            train_examples.append(example)

textcat.initialize(lambda: train_examples, nlp=nlp)
epochs = 20
with nlp.select_pipes(enable="textcat"):
    optimizer = nlp.resume_training()
    for i in range(epochs):
        for example in train_examples:
            nlp.update([example], sgd=optimizer)

text = "Donald Duck want to transferring funds all his money in bank of America. How he can increase his bank balance?"
doc = nlp(text)
intent_scores = doc.cats
predicted_intent = max(intent_scores, key=intent_scores.get)

print("Text:", text)
print("Predicted Intent:", predicted_intent)
print("Intent Scores:", intent_scores)