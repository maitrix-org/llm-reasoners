import os
import random
import openai
import numpy as np
import yaml
import json
import hashlib
from tarski.io import PDDLReader
from tarski.syntax.formulas import *


class Instance_Generator():
    def __init__(self, config_file):
        self.data = None
        self.read_config(config_file)
        self.instances_template = f"./instances/{self.data['domain_name']}/{self.data['instance_dir']}/{self.data['instances_template']}"
        self.label_json = f"./instances/{self.data['domain_name']}/{self.data['domain_name']}_all_labels.json"
        self.hashset = set()
        self.plan_hashset = set()
        if os.path.exists(self.label_json):
            with open(self.label_json, 'r') as file:
                self.all_labels = json.load(file)
        else:
            self.all_labels = {}
        
        os.makedirs(f"./instances/{self.data['domain_name']}/{self.data['instance_dir']}/", exist_ok=True)


    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            self.data = yaml.safe_load(file)

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

    def add_existing_files_to_hash_set(self):
        em = []
        count = 0
        for i in os.listdir(f"./dataset/{self.data['domain_name']}/{self.data['domain']}/"):
            try:
                f = open(f"./dataset/{self.data['domain_name']}/{self.data['domain']}/instance-{count}.pddl", "r")
            except:
                em.append(count)
                count+=1
                continue
            pddl = f.read()
            if pddl:
                to_add = self.convert_pddl(pddl)
                self.hashset.add(to_add)
            else:
                em.append(count)
            count+=1


        length = len(self.hashset)
        try:
            for i in os.listdir(f"./instances/{self.data['domain']}/"):
                f = open(f"./instances/{self.data['domain']}/{i}", "r")
                pddl = f.read()
                to_add = self.convert_pddl(pddl)
                if to_add in self.hashset:
                    print("OOPS")
                self.hashset.add(to_add)
        except FileNotFoundError:
            pass
        return length, em

    def plan_length_validity(self, domain, instance):        
        fast_downward_path = os.getenv("FAST_DOWNWARD")
        # print(fast_downward_path)
        # Remove > /dev/null to see the output of fast-downward
        assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        cmd = f"timeout 20s {fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null"
        os.system(cmd)
        plan_file = "sas_plan"
        try:
            with open(plan_file) as f:
                plan = [line.rstrip() for line in f][:-1]
                readable_plan = '\n'.join(plan)
            os.remove(plan_file)
            print(len(plan))
            return (True, readable_plan) if len(plan) >=3 else (False, "")
        except FileNotFoundError:
            print("No plan found")
            return (False, "")
        

    def convert_pddl(self, pddl):
        init = []
        goal = []
        init_check=False
        goal_check=False
        for i in pddl.split('\n'):
            if 'init' in i:
                init_check = True
                continue
            elif 'goal' in i:
                goal_check=True
                init_check=False
                continue
            to_append = i.replace("(","").replace(")","")
            if to_append and 'and' not in to_append:
                if init_check:
                    init.append(to_append)
                elif goal_check:
                    goal.append(to_append)
        pddl_to_hash = ','.join(sorted(init)+sorted(goal))
        hash_of_instance = hashlib.md5(pddl_to_hash.encode('utf-8')).hexdigest()

        return hash_of_instance


    def gen_goal_directed_instances(self, n_instances, max_objs):
        if self.data['domain'] == 'blocksworld':
            self.gen_goal_directed_instances_blocksworld(n_instances, max_objs)
            
        elif self.data['domain'] == 'logistics':
            self.gen_goal_directed_instances_logistics(n_instances)
        else:
            raise NotImplementedError

    def gen_goal_directed_instances_blocksworld(self, n_instances, max_objs=5):
        if n_instances:
            n = n_instances
        else:
            n = self.data['n_instances'] + 1
        n_objs = range(3, max_objs+1)
        CWD = os.getcwd()
        CMD = "./blocksworld 4 {}"
        start, missing = self.add_existing_files_to_hash_set()

        os.chdir("pddlgenerators/blocksworld/")
        instance_file = f"{CWD}/{self.instances_template}"
        domain = f"{CWD}/instances/{self.data['domain_file']}"
        print(missing)
        c = missing.pop() if missing else start
        for obj in n_objs:
            print(f'==================== Number of blocks {obj} ====================')
            count = 0
            cmd_exec = CMD.format(obj)
            if c>n:
                break
            while count<50:
                with open(instance_file.format(c), "w+") as fd:
                    pddl = os.popen(cmd_exec).read()
                    hash_of_instance = self.convert_pddl(pddl)
                    # hash_of_instance = hashlib.md5(pddl.encode('utf-8')).hexdigest()
                    if hash_of_instance in self.hashset:
                        # print("[-]: Same instance, skipping...")
                        count+=1
                        continue
                    count=0
                    self.hashset.add(hash_of_instance)
                    fd.write(pddl)

                inst_to_parse = instance_file.format(c)
                if self.instance_ok(domain, inst_to_parse):
                    if missing:
                        c = missing.pop()
                    else:
                        if c<start:
                            c=start
                        else:
                            c += 1
                    print(f"[+]: Instance created. Total instances: {c}")
                else:
                    # print("[-]: Instance not valid.")
                    self.hashset.remove(hash_of_instance)
                    os.remove(inst_to_parse)
                    continue


        print(f"[+]: A total of {c} instances have been generated")
        os.chdir(CWD)

    def add_objects_logistics(self, cities, airplanes, packages, city_size):
        
        if airplanes+1>cities:
            airplanes = 1
            if packages+1>city_size*cities:
                packages = 1
                if city_size+1>3:
                    city_size=1
                    if cities+1>5:
                        cities=1
                    else:
                        cities+=1
                else:
                    city_size+=1
            else:                
                packages+=1
        else:
            airplanes+=1
        # print('extended objects')
        return cities, airplanes, packages, city_size

    def gen_goal_directed_instances_logistics(self, n_instances):
        
        CWD = os.getcwd()
        CMD = "./pddlgenerators/logistics/logistics -a {} -c {} -s {} -p {}"
        all_instances = []
        
        instance_file = f"{CWD}/{self.instances_template}"
        domain = f"{CWD}/instances/{self.data['domain_name']}/{self.data['domain_file']}"
        start, missing = self.add_existing_files_to_hash_set()
        # os.chdir("pddlgenerators/logistics/")
        c = missing.pop() if missing else start
        
        all_instances = []
        if n_instances:
            n = n_instances
        else:
            n = self.data['n_instances'] + 1
        global_count = 0
        cities = 2
        airplanes = 1
        city_size = 1
        packages = 1
        while True:
            if c>n:
                break
            instance_label = f"{cities}-{airplanes}-{city_size}-{packages}"
            if instance_label in all_instances:
                global_count+=1
                continue
            print("[INFO]: Instance label: ", instance_label)
            cmd_exec = CMD.format(airplanes, cities, city_size, packages)
            count = 0
            while count<50:
                with open(instance_file.format(c), "w+") as fd:
                    pddl = os.popen(cmd_exec).read()
                    # print(pddl)
                    hash_of_instance = self.convert_pddl(pddl)
                    if hash_of_instance in self.hashset:
                            # print(f"[-]: Same instance {count}, skipping...")
                            count+=1
                            continue
                    count=0
                    self.hashset.add(hash_of_instance)
                    fd.write(pddl)

                inst_to_parse = instance_file.format(c)
                print(self.plan_length_validity(domain, inst_to_parse))
                isvalid, plan = self.plan_length_validity(domain, inst_to_parse)
                if isvalid:
                    # print("Valid plan")
                    plan_hash = plan
                    if plan_hash in self.plan_hashset:
                        print("[-]: Same plan, skipping...")
                        self.hashset.remove(hash_of_instance)
                        os.remove(inst_to_parse)
                        continue
                    self.plan_hashset.add(plan_hash)
                    if instance_label in self.all_labels:
                        self.all_labels[instance_label].append(c)
                    else:
                        self.all_labels[instance_label] = [c]
                    if missing:
                        c = missing.pop()
                    else:
                        if c<start:
                            c=start
                        else:
                            c += 1
                    
                    # print(f"[+]: Instance created. Total instances: {c}")
                else:
                    # print("[-]: Instance not valid.")
                    self.hashset.remove(hash_of_instance)
                    os.remove(inst_to_parse)
                    continue
            
            
            all_instances.append(instance_label)
            with open(f"{self.data['domain_name']}_all_labels.json", 'w') as file:
                json.dump(self.all_labels, file)
            cities, airplanes, packages, city_size = self.add_objects_logistics(cities, airplanes, packages, city_size)


        print(f"[+]: A total of {c} instances have been generated")
        os.chdir(CWD)

class GeneralizationInstanceGenerator:
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

    def t5_gen_generalization_instances(self):
        if self.data['domain_name'] == "blocksworld":
            self.t5_gen_generalization_instances_blocksworld()
        elif self.data['domain_name'] == "logistics":
            self.t5_gen_generalization_instances_logistics()

    def t5_gen_generalization_instances_blocksworld(self, n_instances):
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
        if n_instances:
            n = n_instances
        else:
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

    def t5_gen_generalization_instances_logistics(self):
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

        if n_instances:
            n = n_instances
        else:
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

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, default='blocksworld')
    parser.add_argument('--is_generalization', action='store_true', help='generate generalization instances')
    parser.add_argument('--n_instances', type=int, default=0)
    parser.add_argument('--max_blocks', type=int, default=5, help='max number of blocks in blocksworld')

    args = parser.parse_args()
    config_file = args.config
    is_generalization = args.is_generalization
    n_instances = args.n_instances
    max_blocks = args.max_blocks
    config_file = f'configs/{config_file}.json'
    assert os.path.exists(config_file), f'[-] Config file {config_file} does not exist'
    if is_generalization:
        ig = GeneralizationInstanceGenerator(config_file)
        ig.t5_gen_generalization_instances(n_instances)
    else:
        ig = Instance_Generator(config_file)
        ig.gen_goal_directed_instances(n_instances, max_blocks)
            
            