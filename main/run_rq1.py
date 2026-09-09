import sys
import toml

from termcolor import colored

from main import parse_args
from utils.other_utils import cleanup

if __name__ == '__main__':
    # load sys.argv from toml file
    toml_path = f'../configs/rq1.toml'
    config = toml.load(toml_path)
    argv = config["args"]

    # get list for batch experiments
    method_list = config["methods"]
    model_list = config["models"]
    cand_types = config["cand_types"]

    print(colored(f'Methods list is {method_list}', 'red', attrs=['bold', 'underline']), )

    for model in model_list:
        argv.extend(["-model", model])

        for cand_type in cand_types:
            argv.extend(["-cand_type", cand_type])

            for method in method_list:
                argv.extend(["-method", method])

                sys.argv = argv

                # run experiment
                result = parse_args()
                print("\n")
                cleanup()
