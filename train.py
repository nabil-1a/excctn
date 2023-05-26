import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

#Choose trainer file
from trainer.trainer_excctn import Trainer
from utils.utils import initialize_config
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='EXCCTN_1')
parser.add_argument("-C", "--config", required=True, type=str,
                    help="configuration for training (*.json).")
parser.add_argument('-D', '--device', default=None, type=str,
                    help="GPUs visible, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true",
                    help="Resume from checkpoint.")
args = parser.parse_args()

#no devices are visible in cluster environment, so set to "0"
if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(config, resume):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    def pad_to_longest(batch):
        mix_list = []
        clean_list = []
        names = []
        n_frames_list = []

        for mixture, clean, n_frames, name in batch:
            mix_list.append(torch.tensor(mixture).reshape(-1, 1))
            clean_list.append(torch.tensor(clean).reshape(-1, 1))
            n_frames_list.append(n_frames)
            names.append(name)

        mix_list = pad_sequence(mix_list).squeeze(2).permute(1, 0)
        clean_list = pad_sequence(clean_list).squeeze(2).permute(1, 0)

        return mix_list, clean_list, n_frames_list, names

    #load training and validation datasets
    train_dataset = initialize_config(config["train_dataset"])
    tr_loader = DataLoader(
        shuffle=config["train_dataloader"]["shuffle"],
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        collate_fn=pad_to_longest,
        drop_last=True
    )

    cv_dataset = initialize_config(config["validation_dataset"])
    cv_loader = DataLoader(
        dataset=cv_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"],
        collate_fn=pad_to_longest,
        shuffle=config["validation_dataloader"]["shuffle"]
    )
    model = initialize_config(config["model"])
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], 0.999)
    )
    loss_function = initialize_config(config["loss_function"])
    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_dataloader=tr_loader,
        validation_dataloader=cv_loader
    )
    trainer.train()

if __name__ == '__main__':
    config = json.load(open(args.config))
    config["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]
    config["train_config_path"] = args.config
    main(config, resume=args.resume)
