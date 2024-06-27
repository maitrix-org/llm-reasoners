import os

# Src location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Keywords for internal dictionary based
# representation of classical planning models
# Expected domain dictionary format
# domain:
#   action_name:
#       params: list
#       pos_prec: set()
#       neg_prec: set()
#       adds: set()
#       dels: set()
#       conditional_adds: list()
#       conditional_dels: list()
# instance:
#   init: set()
#   goal: set()
DOMAIN = "domain"
PARARMETERS = "params"
POS_PREC = "pos_prec"
NEG_PREC = "neg_prec"
ADDS = "adds"
INCREMENT = "increase"
COND_ADDS = "conditional_adds"
COND_DELS = "conditional_adds"
DELS = "dels"
INSTANCE = "instance"
INIT = "init"
GOAL = "goal"
PREDICATES = "pred"
FUNCTIONS  = "functions"
FUNCTIONAL = "functional"
HIERARCHY = "hierarchy"
ANCESTORS = "ancestors"
IMM_PARENT = "imm_parent"
CONSTANTS = "constants"
METRIC = "metric"
# A hack to get arround multiple goals
GOAL_ACHIEVED = "goal_achieved"
GOAL_ACT = "goal-act"
COST = "cost"