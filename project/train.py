import argparse
import json
from pathlib import Path

import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(config, payload):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.get("checkpoint_name", "checkpoint.pt")
    torch.save(payload, checkpoint_path)
    print(checkpoint_path)


def run_sft(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = config.get("input_dim", 8)
    batch_size = config.get("batch_size", 4)
    steps = config.get("steps", 10)

    model = torch.nn.Linear(input_dim, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))

    for _ in range(steps):
        x = torch.randn(batch_size, input_dim, device=device)
        y = x.sum(dim=1, keepdim=True)
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_checkpoint(
        config,
        {
            "mode": "sft",
            "model_state": model.state_dict(),
            "steps": steps,
        },
    )


def run_rl(config):
    steps = config.get("steps", 10)
    score = 0.0

    for _ in range(steps):
        action = torch.randn(1)
        reward = -action.pow(2).item()
        score += reward

    save_checkpoint(
        config,
        {
            "mode": "rl",
            "steps": steps,
            "score": score,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_min.json")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = config.get("mode", "sft")

    if mode == "sft":
        run_sft(config)
        return
    if mode == "rl":
        run_rl(config)
        return

    raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    main()
