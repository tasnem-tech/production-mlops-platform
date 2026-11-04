import json
from src.model import train_and_evaluate
if __name__ == '__main__':
    _, metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))
