from .parser_new import parse_model
from .writer_new import ModelWriter

class Parser_PDDL:
    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file
    def parse_PDDL(self):
        model = parse_model(self.domain_file, self.problem_file)
        return model
    def write_PDDL(self, model):
        writer = ModelWriter(model)
        writer.write_model()