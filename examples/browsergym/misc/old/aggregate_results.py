import os

exp_containing_dir = "./results"
for exp in os.listdir(exp_containing_dir):
  exp_dir = os.path.join(exp_containing_dir, exp)

  exp_status = open(os.path.join(exp_dir, "status.txt"), "r").readlines()

  success = 0; total = 0
  for line in exp_status:
    # print(line)
    if line.strip().split(" ")[1] == "True":
      success += 1
    total += 1

  print(f"{exp}: {success/total} - {success} - {total}")

