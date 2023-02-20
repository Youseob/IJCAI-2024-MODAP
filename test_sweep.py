import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr_initial', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default="walker2d-random")

args    = parser.parse_args()

# 입력받은 인자값 출력
print(args.dataset)