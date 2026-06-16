import json
import os

query = """You are a world-class Cypher expert specializing in Neo4j graph databases.

    Given a graph schema and a user's natural language question, generate the most accurate Cypher query possible.

    Rules:
    - Use only entities and relationships present in the schema.
    - Never hallucinate labels, relationship types, or properties.
    - Respect relationship directions as defined in the schema.
    - Use meaningful variable names.
    - Return only executable Cypher.
    - Do not include explanations, markdown, comments, or additional text.
"""

def format_training_data(data, tokenizer):
    messages = [
        {
            "role": "system",
            "content": query,
        },
        {
            "role": "user",
            "content": f"SCHEMA: {data['schema']},\nQUESTION: {data['question']}"
        },
        {
            "role": "assistant",
            "content": data["cypher"]
        }
    ]
    result = tokenizer.apply_chat_template(messages, tokenize=False)
    return result

def prepare_data(data, tokenizer):
        datapath = "./../dataset_prepared"
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        for category in data:
            path = f"{datapath}/{category}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for example in data[category]:
                    f.write(json.dumps(format_training_data(example, tokenizer), ensure_ascii=False) + "\n")

