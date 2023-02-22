import os


dataset = str(os.environ.get('dataset', "None"))
learning_rate = float(os.environ.get('learning_rate', 0))
temp = float(os.environ.get('temp', 0))

print(f"{dataset}-{learning_rate}-{temp}")