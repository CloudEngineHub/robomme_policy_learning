from mme_vla_suite.remote_evaluation.server import PolicyServer
from mme_vla_suite.policies.policy_config import create_trained_policy
from mme_vla_suite.remote_evaluation.policy import MyPolicy_for_CVPR_Challenge
from pathlib import Path
from mme_vla_suite.training.config import get_config
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a policy for the CVPR challenge.")

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("my_cool_model"),
        help="Path to the checkpoint directory (default: %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = create_trained_policy(
        train_config=get_config("mme_vla_suite"),
        checkpoint_dir=args.checkpoint_dir,
        seed=7,
    )

    policy = MyPolicy_for_CVPR_Challenge(model=model)
    server = PolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()