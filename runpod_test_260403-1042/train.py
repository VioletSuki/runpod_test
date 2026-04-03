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


def build_train_params(config):
    excluded_keys = {"output_dir", "checkpoint_name"}
    return {key: value for key, value in config.items() if key not in excluded_keys}


def emit_result(epoch, train_params, success):
    print(
        "OPENCLAW_RESULT_JSON="
        + json.dumps(
            {
                "epoch": epoch,
                "train_params": train_params,
                "success": success,
            },
            separators=(",", ":"),
        )
    )


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
    return 1


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
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_min.json")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = config.get("mode", "sft")
    train_params = build_train_params(config)

    try:
        if mode == "sft":
            epoch = run_sft(config)
        elif mode == "rl":
            epoch = run_rl(config)
        else:
            raise ValueError(f"unsupported mode: {mode}")

        emit_result(epoch, train_params, True)
    except Exception:
        emit_result(0, train_params, False)
        raise


if __name__ == "__main__":
    main()
