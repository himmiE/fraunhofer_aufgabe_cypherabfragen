import json

from src.load_data import load_data, load_prepared_data


def get_results(model_name):
    from src.load_base_model import load_model

    tokenizer, model = load_model(model_name)
    examples = load_prepared_data("test")["messages"]

    for example in examples:
        ground_truth = example[-1]["content"]
        input_text = tokenizer.apply_chat_template(example, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
        prediction = tokenizer.decode(outputs[0])

        path = f"./../results/{model_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

def prepare_eval_data():
    examples = load_prepared_data("test")["messages"]

def eval():
    get_results("base_model_gpu")
    get_results("new_model_2")
    pass

