from sentence_transformers import SentenceTransformer, InputExample, losses, models
from datasets import Dataset
import json

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
import os
import dotenv
dotenv.load_dotenv()

# Load base embedding model
model = SentenceTransformer("dwzhu/e5-base-4k")

dataset = load_dataset("Bossologist/embed-ft-phase-a-data")

train_examples: list[InputExample] = []
for example in dataset["train"]:
    train_examples.append(InputExample(texts=[example["event_a"], example["event_b"]], label=example["label"]))

val_examples: list[InputExample] = []
for example in dataset["validation"]:
    val_examples.append(InputExample(texts=[example["event_a"], example["event_b"]], label=example["label"]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Create the evaluator
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=dataset['validation']['event_a'],
    sentences2=dataset['validation']['event_b'],
    scores=dataset['validation']['label'],
    name='sts-validation' # A name for the evaluation results
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=2,
    warmup_steps=100,
    evaluation_steps=500,
    output_path='./fine_tuned_model_with_val',
    save_best_model=True # Now saves the model with the best validation score
)
model.push_to_hub("Bossologist/e5-base-4k-phase-a", token=os.environ["HF_KEY"])