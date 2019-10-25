from __future__ import print_function
from wine_quality.train import Trainer
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="sklearn", required=False)
    parser.add_argument("--data_path", dest="data_path", help="data_path", required=True)
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=None, type=int)
    parser.add_argument("--max_leaf_nodes", dest="max_leaf_nodes", help="max_leaf_nodes", default=None, type=int)
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="none")
    parser.add_argument("--registered_model", dest="registered_model", help="Registered model name")
    args = parser.parse_args()
    trainer = Trainer(args.experiment_name, args.data_path,args.run_origin, args.registered_model)
    trainer.train( args.max_depth, args.max_leaf_nodes)
