import argparse
import json
import os
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from utils.stft import STFT
from utils.utils import initialize_config
from utils.conv_stft import ConvSTFT, ConviSTFT


def main(config, epoch):
    root_dir = Path(config["exp_dir"]) #/ config["name"]
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"
	transform = ConvSTFT(400, 100, 512)
	inverse = ConviSTFT(400, 100, 512)

    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )

    model = initialize_config(config["model"])
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    # stft = STFT(
    #    filter_length=512,
    #    hop_length=100
    # ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()

    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_model_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    for i, (mixture, _, _, names) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]

        print("Generating STFT")
        # mixture_D = stft.transform(mixture)
		mixture_D = transform(mixture)

        print("Starting Enhancement")
        enhanced_real, enhanced_imag = model(mixture_D)
		enhanced_D = torch.cat([enhanced_real, enhanced_imag], 1)
        # enhanced_D = enhanced_D.detach().cpu().unsqueeze(0)
		enhanced = self.inverse(enhanced_D)
        # enhanced = stft.inverse(enhanced_D)
        enhanced = enhanced.detach().cpu().squeeze().numpy()

        sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    parser.add_argument("-C", "--config", type=str, required=True,
                        help="Specify the configuration file for enhancement (*.json).")
    parser.add_argument("-E", "--epoch", default="best",
                        help="Checkpoint for enhancement ('best', 'latest' or specific epoch; default: 'best')")
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    main(config, args.epoch)
