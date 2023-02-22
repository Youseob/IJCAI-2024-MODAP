import os
import sys

jobs=[]
for add_value in [True, False]:
    for prior_reg in [10, 100, 50]:
        for temp in [10, 100]:
            for belief_mode in ["kl-reg", "softmax"]:
                jobs.append((add_value, prior_reg, temp, belief_mode))

(add_value, prior_reg, temp, belief_mode) = jobs[int(sys.argv[-1])]

os.environ["prior_reg"] = str(prior_reg)
os.environ["add_value"] = str(add_value)
os.environ["temp"] = str(temp)
os.environ["belief_mode"] = str(belief_mode)


os.system("bash launch_adv.sh 110 100")
# dataset = str(os.environ.get('dataset', "None"))
# learning_rate = float(os.environ.get('learning_rate', 0))
# temp = float(os.environ.get('temp', 0))

# print(f"{dataset}-{learning_rate}-{temp}")