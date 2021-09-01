from config import Cfg
from utils import fullseed
from train_functions import run

fullseed(Cfg.seed)

run(Cfg)