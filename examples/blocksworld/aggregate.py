import fire
def main(prefix):
    cc, ff = 0, 0
    for step in range(2, 13, 2):
        path = f"logs/{prefix}_blocksworld_cot_v1_step{step}/result.log"
        c, f = 0, 0
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "correct=False" in line:
                    f += 1
                elif "correct=True" in line:
                    c += 1
        print(f"step {step}: correct={c}, failed={f}")
        cc += c
        ff += f
    print(f"total: correct={cc}, failed={ff}, accuracy={cc/(cc+ff)}")
    return 0

if __name__ == '__main__':
    fire.Fire(main)