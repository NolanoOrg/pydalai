import argparse
import copy
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=tuple, default=("7B",))

    args = parser.parse_args()

    num = {
        "7B": 1,
        "13B": 2,
        "30B": 4,
        "65B": 8
    }

    files = ["checklist.chk", "params.json"]

    for model in args.models:
        cur_files = copy.deepcopy(files)
        for i in range(num[model]):
            cur_files.append(f"consolidated.0{i}.pth")

        if not os.path.exists("llama.cpp/models"):
            os.makedirs("llama.cpp/models")

        if not os.path.exists(f"llama.cpp/models/{model}"):
            os.makedirs(f"llama.cpp/models/{model}")

        for cur_file in cur_files:
            if not os.path.exists(f"llama.cpp/models/{model}/{cur_file}"):
                os.system(f"wget https://agi.gpt4.org/llama/LLaMA/{model}/{cur_file} -P llama.cpp/models/{model}")

        print(f"Downloaded {model} model")

    tokenizer_files = ["tokenizer_checklist.chk", "tokenizer.model"]

    for cur_file in tokenizer_files:
        if not os.path.exists("llama.cpp/models"):
            os.makedirs("llama.cpp/models")

        if not os.path.exists(f"llama.cpp/models/{cur_file}"):
            os.system(f"wget https://agi.gpt4.org/llama/LLaMA/{cur_file} -P llama.cpp/models")

    print("Downloaded tokenizer")