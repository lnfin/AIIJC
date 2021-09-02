from config import Cfg
from utils import set_seed
from train_functions import run

set_seed(Cfg.seed)

run(Cfg)
