import wandb
from main import train_with_wandb
from argparse import ArgumentParser

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("sweep_id")
    args = parser.parse_args()
    wandb.agent(args.sweep_id, function=train_with_wandb, project='absa-enterpret', entity='saumb7800')