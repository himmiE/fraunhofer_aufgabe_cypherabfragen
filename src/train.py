import os

from src.load_base_model import load_model
from src.load_data import load_data, load_prepared_data

from trl import SFTTrainer
from transformers import TrainingArguments

from src.query_making import prepare_data


def train(name="smoke_test"):
    if not name:
        name = "smollm-cypher"

    path = "./../model/" + name

    if os.path.exists(path):
        return

    tokenizer, model = load_model("base-model")
    data = load_prepared_data("train")

    training_args = TrainingArguments(
        output_dir=path,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=training_args
    )

    trainer.train()

    trainer.save_model(path)
    tokenizer.save_pretrained(path)