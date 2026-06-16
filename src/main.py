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




train()
#init()
#inference()

