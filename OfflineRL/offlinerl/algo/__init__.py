from loguru import logger
import warnings

warnings.filterwarnings('ignore')


from offlinerl.config.algo import cql_config, plas_config, mopo_config, moose_config, bcqd_config, bcq_config, bc_config, crr_config, combo_config, bremen_config, maple_config, \
        bayrl_config, bayrl_cvar_config, model_analysis_config, bayrl_calib_config, bayrl_v2_config, \
        adv_bayrl_config, adv_bayrl_v2_config, adv_bayrl_v3_config, pessi_config

from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import cql, plas, bcqd, bcq, bc, crr
from offlinerl.algo.modelbase import mopo, moose, combo, bremen, maple, bayrl, bayrl_cvar, model_analysis, bayrl_calib,\
                                    adv_bayrl, adv_bayrl_v2, adv_bayrl_v3, pessimistic_bayrl, bayrl_v2, pessimistic_bayrl_v2

from offlinerl.algo.modelbase import maple_div
from offlinerl.config.algo import maple_div_config

algo_dict = {
    'bc' : {"algo" : bc, "config" : bc_config},
    'bcq' : {"algo" : bcq, "config" : bcq_config},
    'bcqd' : {"algo" : bcqd, "config" : bcqd_config},
    'combo' : {"algo" : combo, "config" : combo_config},
    "cql" : {"algo" : cql, "config" : cql_config},
    "crr" : {"algo" : crr, "config" : crr_config},
    "plas" : {"algo" : plas, "config" : plas_config},
    'moose' : {"algo" : moose, "config" : moose_config},
    'mopo': {"algo" : mopo, "config": mopo_config},
    'bremen' : {"algo" : bremen, "config" : bremen_config},
    'maple': {'algo':maple , 'config':maple_config},
    'bayrl': {'algo':bayrl , 'config':bayrl_config},
    'bayrl_cvar': {'algo':bayrl_cvar , 'config':bayrl_cvar_config},
    'model_analysis': {'algo': model_analysis, 'config': model_analysis_config},
    'bayrl_calib': {'algo': bayrl_calib, 'config': bayrl_calib_config},
    'adv_bayrl': {'algo': adv_bayrl, 'config': adv_bayrl_config},
    'adv_bayrl_v2': {'algo': adv_bayrl_v2, 'config': adv_bayrl_v2_config},
    'adv_bayrl_v3': {'algo': adv_bayrl_v3, 'config': adv_bayrl_v3_config},
    'pessi_bayrl' : {'algo': pessimistic_bayrl, 'config': pessi_config},
    'pessi_bayrl_v2' : {'algo': pessimistic_bayrl_v2, 'config': pessi_config},
    'bayrl_v2' : {'algo' : bayrl_v2, 'config': bayrl_v2_config},
    'maple_div': {'algo': maple_div , 'config':maple_div_config},
}

def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info('Use {} algorithm!', algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]
    
    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)
    
    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer
    
    return algo_init, algo_trainer, algo_config
    
    