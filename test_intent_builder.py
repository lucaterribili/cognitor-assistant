import json

from classes.DatasetGenerator import DatasetGenerator

with open("data.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

dataset_generator = DatasetGenerator(data)
dataset_generator.generate_nlu()