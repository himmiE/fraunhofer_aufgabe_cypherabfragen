import os

from src.load_base_model import load_model
from src.load_data import load_data, load_prepared_data

from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

from src.query_making import prepare_data




def train(name="new_model_2"):
    if not name:
        name = "smollm-cypher"

    path = "./../model/" + name

    if os.path.exists(path):
        print("modelpath exists")
        return

    tokenizer, model = load_model("base-model_gpu")
    data_train = load_prepared_data("train")
    data_val = load_prepared_data("val")

    training_args = TrainingArguments(
        output_dir=path,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,

        save_total_limit=2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    trainer.save_model(path)
    tokenizer.save_pretrained(path)