import os
import random
import openai
import numpy as np
import hashlib
from tarski.io import PDDLReader
from tarski.syntax.formulas import *

openai.api_key = os.environ["OPENAI_API_KEY"]
random.seed(10)

from .llm_utils import *
from .pddl_to_text import *
from .text_to_pddl import *
from .task_utils import *
import yaml

import os
import random
import openai
import numpy as np
import hashlib
import yaml
class LogisticsGenerator:
    def __init__(self, config_file):
        random.seed(10)
        self.data = self.read_config(config_file)
        self.instances_template_t5 = f"./instances/{self.data['generalized_instance_dir']}/{self.data['instances_template']}"
        
        self.hashset = set()
        self.instances = []
        os.makedirs(f"./instances/{self.data['generalized_instance_dir']}/", exist_ok=True)
    
    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def add_existing_files_to_hash_set(self, instance_dir=None):
        for i in os.listdir(f"./instances/{instance_dir}/"):
            f = open(f"./instances/{instance_dir}/" + i, "r")
            pddl = f.read()
            self.hashset.add(hashlib.md5(pddl.encode('utf-8')).hexdigest())
        return len(self.hashset)
    

    def t5_gen_generalization_instances(self):
        def gen_instance(init, goal, objs):
            text = "(define (problem LG-generalization)\n(:domain logistics-strips)"
            text += "(:objects " + " ".join(objs) + ")\n"
            text += "(:init \n"
            text += "\n".join(init)
            text += "\n"
            text += ")\n(:goal\n(and\n"
            text += "\n".join(goal)
            text += "\n"
            text += ")))"
            return text

        n = self.data['n_instances'] + 1
        start = self.add_existing_files_to_hash_set(self.data['generalized_instance_dir']) + 1
        print("[+]: Making generalization instances for logistics")
        c = start
        while c<n:
            cities = list(range(random.randint(1, 3)))
            locations = list(range(random.randint(3, 10)))
            packages = list(range(random.randint(2, len(locations))))
            random.shuffle(cities)
            random.shuffle(locations)
            random.shuffle(packages)
            # print(f"[+]: Generating instance {c} with {len(cities)} cities, {len(locations)} locations, {len(packages)} packages")
            init = []
            goal = []
            objs = []
            airports = {}
            for city in cities:
                init.append(f"(CITY c{city})")
                init.append(f"(TRUCK t{city})")
                init.append(f"(AIRPLANE a{city})")
                objs+=[f"c{city}", f"t{city}", f"a{city}"]
                pack_done = 0
                for location in locations:
                    init.append(f"(LOCATION l{city}-{location})")
                    init.append(f"(in-city l{city}-{location} c{city})")
                    objs.append(f"l{city}-{location}")
                    if pack_done < len(packages):
                        to_mul = city*len(packages)
                        init.append(f"(OBJ p{to_mul+packages[pack_done]})")
                        objs.append(f"p{to_mul+packages[pack_done]}")
                        if pack_done == 0:
                            init.append(f"(at p{to_mul+packages[pack_done]} l{city}-{location})")
                            init.append(f"(at t{city} l{city}-{location})")
                        else:
                            init.append(f"(at p{to_mul+packages[pack_done]} l{city}-{location})")
                            goal.append(f"(at p{to_mul+packages[pack_done-1]} l{city}-{location})")
                        pack_done += 1
                airports[city] = (location, packages[pack_done-1])
            for city, v in airports.items():
                location, package = v
                init.append(f"(AIRPORT l{city}-{location})")
                init.append(f"(at a{city} l{city}-{location})")
                if len(cities) > 1:
                    #pick a city to fly to which is not the current city
                    fly_to = random.choice(list(airports.keys()))
                    while fly_to == city:
                        fly_to = random.choice(list(airports.keys()))
                    to_mul = city*len(packages)
                    goal.append(f"(at p{to_mul+package} l{fly_to}-{airports[fly_to][0]})")

            instance = gen_instance(init, goal, objs)

            if hashlib.md5(instance.encode('utf-8')).hexdigest() in self.hashset:
                print("[-] INSTANCE ALREADY IN SET, SKIPPING")
                continue

            with open(self.instances_template_t5.format(c), "w+") as fd:
                fd.write(instance)
            # print(f"[+] Instance {c} generated")
            c+=1



class BWGenerator:
    def __init__(self, config_file):
        self.data = self.read_config(config_file)
        self.instances_template = f"./instances/{self.data['instance_dir']}/{self.data['instances_template']}"
        self.instances_template_t5 = f"./instances/{self.data['generalized_instance_dir']}/{self.data['instances_template']}"

        self.hashset = set()
        os.makedirs(f"./instances/{self.data['instance_dir']}/", exist_ok=True)
        os.makedirs(f"./instances/{self.data['generalized_instance_dir']}/", exist_ok=True)

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
        
    def instance_ok(self, domain, instance):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        reader.parse_instance(instance)
        if isinstance(reader.problem.goal, Tautology):
            return False
        elif isinstance(reader.problem.goal, Atom):
            if reader.problem.goal in reader.problem.init.as_atoms():
                return False
        else:
            if (all([i in reader.problem.init.as_atoms() for i in reader.problem.goal.subformulas])):
                return False
        return True

    def add_existing_files_to_hash_set(self, inst_dir):
        for i in os.listdir(f"./instances/{inst_dir}/"):
            f = open(f"./instances/{inst_dir}/" + i, "r")
            pddl = f.read()
            self.hashset.add(hashlib.md5(pddl.encode('utf-8')).hexdigest())
        return len(self.hashset)

    def t1_gen_goal_directed_instances(self):
        n = self.data['n_instances'] + 2
        n_objs = range(4, len(self.data["encoded_objects"]) + 1)
        ORIG = os.getcwd()
        CMD = "./blocksworld 4 {}"
        start = self.add_existing_files_to_hash_set(self.data['instance_dir'])

        os.chdir("pddlgenerators/blocksworld/")
        instance_file = f"{ORIG}/{self.instances_template}"
        domain = f"{ORIG}/instances/{self.data['domain_file']}"
        c = start
        for obj in n_objs:
            cmd_exec = CMD.format(obj)
            for i in range(1, n):
                with open(instance_file.format(c), "w+") as fd:
                    pddl = os.popen(cmd_exec).read()
                    hash_of_instance = hashlib.md5(pddl.encode('utf-8')).hexdigest()
                    if hash_of_instance in self.hashset:
                        print("[+]: Same instance, skipping...")
                        continue
                    self.hashset.add(hash_of_instance)
                    fd.write(pddl)

                inst_to_parse = instance_file.format(c)
                if self.instance_ok(domain, inst_to_parse):
                    c += 1
                else:
                    print("[-]: Instance not valid")
                    self.hashset.remove(hash_of_instance)
                    os.remove(inst_to_parse)
                    continue
                if c == n:
                    break
            if c == n:
                break

        print(f"[+]: A total of {c} instances have been generated")
        os.chdir(ORIG)

    def t5_gen_generalization_instances(self):
        def gen_instance(objs):
            text = "(define (problem BW-generalization-4)\n(:domain blocksworld-4ops)"
            text += "(:objects " + " ".join(objs) + ")\n"
            text += "(:init \n(handempty)\n"

            for obj in objs:
                text += f"(ontable {obj})\n"

            for obj in objs:
                text += f"(clear {obj})\n"

            text += ")\n(:goal\n(and\n"

            obj_tuples = list(zip(objs, objs[1:]))
            # obj_tuples.reverse() # TODO: this improves considerably Davinci t4

            for i in obj_tuples:
                text += f"(on {i[0]} {i[1]})\n"

            text += ")))"
            return text

        n = self.data['n_instances'] + 2
        objs = self.data['encoded_objects']
        encoded_objs = list(objs.keys())
        start = self.add_existing_files_to_hash_set(self.data['generalized_instance_dir'])

        print("[+]: Making generalization instances for blocksworld")
        for c in range(start, n):
            n_objs = random.randint(3, len(objs))
            random.shuffle(encoded_objs)
            objs_instance = encoded_objs[:n_objs]
            instance = gen_instance(objs_instance)

            if hashlib.md5(instance.encode('utf-8')).hexdigest() in self.hashset:
                print("INSTANCE ALREADY IN SET, SKIPPING")
                continue

            with open(self.instances_template.format(c), "w+") as fd:
                fd.write(instance)





def treat_on(letters_dict, atom):
    terms = atom.subterms
    return f"the {letters_dict[terms[0].name]} block on top of the {letters_dict[terms[1].name]} block"










def validate_plan(domain, instance, plan_file):
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')
    return True if "Plan valid" in response else False












################################################################
# Generate 2 instances each time
# for c in range(1, n, 2):
#     n_objs = random.randint(3, len(data))
#     random.shuffle(encoded_objs)
#     objs_i1 = encoded_objs[:n_objs]
#     objs_i2 = objs_i1.copy()
#     random.shuffle(objs_i2)
#
#     i1 = gen_instance(objs_i1)
#     i2 = gen_instance(objs_i2)
#
#     with open(INSTANCE_FILE.format(c), "w+") as fd:
#         fd.write(i1)
#     with open(INSTANCE_FILE.format(c+1), "w+") as fd:
#         fd.write(i2)

################################################################





def get_cost_gpt_3(gpt3_response):
    lines = [line.strip() for line in gpt3_response.split("\n")]
    flag = True
    for i in range(len(lines)):
        if 'time to execute' in lines[i]:
            flag = False
        if flag:
            continue
        res = [int(i) for i in lines[i].split() if i.isdigit()]
        if len(res) > 0:
            return res[0]
    return 0





def caesar_encode(query):
    key = 5
    alpha = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz']
    new_query = ''
    for i in query:
        if i in alpha[0]:
            new_letter = (alpha[0].find(i) + key) % 26
            new_query += alpha[0][new_letter]
        elif i in alpha[1]:
            new_letter = (alpha[1].find(i) + key) % 26
            new_query += alpha[1][new_letter]
        else:
            new_query += i
    return new_query


def caesar_decode(gpt3_resp):
    key = 5
    alpha = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz']
    new_query = ''
    for i in gpt3_resp:
        if i in alpha[0]:
            new_letter = (alpha[0].find(i) - key) % 26
            new_query += alpha[0][new_letter]
        elif i in alpha[1]:
            new_letter = (alpha[1].find(i) - key) % 26
            new_query += alpha[1][new_letter]
        else:
            new_query += i
    return new_query
