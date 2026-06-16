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

def format_request(query, data):
    result = {"messages": [
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
    ]}
    return result