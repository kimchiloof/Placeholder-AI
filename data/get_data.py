from datasets import load_dataset
from preprocess_data import format_dataset

raw_dataset = load_dataset("naver-clova-ix/cord-v2")
dataset = format_dataset(raw_dataset, 800, 1200)


def view_dataset():
    print(dataset['train'][0]['ground_truth'])


view_dataset()
