import os


dataset = str(os.environ.get('dataset', "None"))
learning_rate = float(os.environ.get('learning_rate', 0.01)
# 입력받은 인자값 출력
print(learning_rate)