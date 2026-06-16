from src.load_base_model import load_model
from src.load_data import load_data
from src.query_making import prepare_data
from src.train import train

def init():
    tokenizer, model = load_model("base-model")
    data = load_data()
    prepare_data(data,tokenizer)

def inference():
    ds = load_data()
    example = ds["test"][0]
    print(example)
    request(example)

def request(data):
    from src.load_base_model import load_model

    content = f"QUESTION: {data['question']} \nSCHEMA: {data['schema']}"

    tokenizer, model = load_model("base-model")

    messages = [{"role": "system",
                 "content": """You are a world-class Cypher expert specializing in Neo4j graph databases.

    Given a graph schema and a user's natural language question, generate the most accurate Cypher query possible.

    Rules:
    - Use only entities and relationships present in the schema.
    - Never hallucinate labels, relationship types, or properties.
    - Respect relationship directions as defined in the schema.
    - Use meaningful variable names.
    - Return only executable Cypher.
    - Do not include explanations, markdown, comments, or additional text.
"""},
                {"role": "user",
                 "content": content}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cpu")
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    print(tokenizer.decode(outputs[0]))


train()
#init()
#inference()

