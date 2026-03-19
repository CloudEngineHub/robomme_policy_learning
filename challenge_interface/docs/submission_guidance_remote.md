## RoboMME Challenge Guide: Remote API Submission

This document explains how to serve your policy **from your own machine** (no Docker image submission) and submit a **public host + port** to EvalAI for CVPR challenge evaluation.

### What you (participant) provide

- **A reachable policy server endpoint**: `<public_host_or_ip>:<port>`
- **One command** to start your policy server (including `--host`, `--port`, and checkpoint path)
- (Optional) **A checkpoint URL** (e.g., Hugging Face) if you want organizers to reproduce locally

### 1) Implement the policy interface

Implement a `Policy` class compatible with the challenge interface.

- Copy the benchmark interface directory (`remote_evaluation/`) into your repo (as done in `src/mme_vla_suite/remote_evaluation`):
  - `https://github.com/RoboMME/robomme_benchmark/tree/main/src/remote_evaluation`
- Override **`infer`** and **`reset`** in your policy implementation (example: `MyPolicy_for_CVPR_Challenge` in `src/mme_vla_suite/remote_evaluation/policy.py`).

### 2) Prepare the serving entrypoint

Copy and adapt the benchmark serving script into your repo:

- Reference: `https://github.com/RoboMME/robomme_benchmark/blob/main/scripts/challenge_serve_policy.py`
- In this repo, the entrypoint lives at `scripts/challenge_serve_policy.py`.

Pick a port (example: `8001`). Your server must listen on an address reachable from outside.

When you submit on EvalAI, provide a command like:

```bash
CUDA_VISIBLE_DEVICES=0 uv run ./scripts/challenge_serve_policy.py \
  --host 0.0.0.0 \
  --port 8001 \
  --checkpoint-dir runs/ckpts/my_cool_model_name/79999
```

Notes:
- If running on a remote machine, you typically want `--host 0.0.0.0` (bind all interfaces).
- Ensure your firewall/security group allows inbound TCP traffic on the chosen port.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/challenge_serve_policy.py \
  --host 0.0.0.0 \
  --port 8001 \
  --checkpoint-dir runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999
```

### 3) Self-check locally with the benchmark eval client (terminal 1)

In a separate environment for RoboMME (see `examples/robomme/readme.md`), run the benchmark eval script against your running server.

Example:

```bash
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python third_party/robomme_benchmark/scripts/challenge_eval_policy.py \
  --host <your_public_ip_or_dns> \
  --port 8001
```

If you are testing from the **same machine** as the server, you can use:

```bash
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python third_party/robomme_benchmark/scripts/challenge_eval_policy.py \
  --host 127.0.0.1 \
  --port 8001
```

### 4) Submit on EvalAI

Submit your **public host/IP** and **port** on EvalAI to join the challenge:

- Challenge link: `<TODO: add EvalAI challenge URL>`

You are responsible for keeping the server reachable during evaluation.

---

### What the organizers will do

1) Connect to your submitted endpoint:

- `<your_public_host_or_ip>:<port>`

2) Run evaluation using the official benchmark eval script:

- `https://github.com/RoboMME/robomme_benchmark/blob/main/scripts/challenge_eval_policy.py`

For example:

```bash
cd robomme_benchmark
uv run ./scripts/challenge_eval_policy.py --host <your_public_ip_or_dns> --port <your_public_port>
```

Keep your endpoint stable during the evaluation window.
