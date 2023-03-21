import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--epoch',          type=int,   default=150)
parser.add_argument('--batch_size',     type=int,   default=128)
parser.add_argument('--lr_initial',     type=float, default=0.1)

args = parser.parse_args()
name = f"{args.epoch}-{args.batch_size}-{args.lr_initial}"
with open(f"/output/{name}", "w") as f:
    f.write("hi")

