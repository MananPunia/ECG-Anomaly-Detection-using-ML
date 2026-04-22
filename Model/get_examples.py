import pandas as pd
import json

df = pd.read_csv('Train&Test/mitbih_test.csv', header=None)
examples = {}
classes = ['N', 'S', 'V', 'F', 'Q']

for i in range(5):
    row = df[df.iloc[:, -1] == i].iloc[0, :-1].tolist()
    examples[classes[i]] = ', '.join([f"{x:.4f}" for x in row])

with open("examples.json", "w") as f:
    json.dump(examples, f)

print("Saved examples.json")
