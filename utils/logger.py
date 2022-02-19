import wandb

class Logger:

    def __init__(self, config, project="playable-environments"):
        self.config = config

        wandb.init(project=project, name=config["logging"]["run_name"], config=config)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def get_wandb(self):
        return wandb

