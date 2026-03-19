## RoboMME Challenge: Docker Submission Guide

This document explains how to package your policy into a Docker image that the organizers can pull and run for CVPR challenge evaluation.

### What you (participant) provide

- **A Docker image** containing your policy server code and all dependencies.
- **A checkpoint location** that the organizers can download (e.g., a Hugging Face repo).
- **One command** to start your policy server inside the container (including port and checkpoint path).

### 1) Implement the policy interface

Implement a `Policy` class compatible with the challenge interface.

- Copy the benchmark interface directory (`remote_evaluation/`) into your repo (as done in `src/mme_vla_suite/remote_evaluation`):
  - `https://github.com/RoboMME/robomme_benchmark/tree/main/src/remote_evaluation`
- Override **`infer`** and **`reset`** in your policy implementation (example: `MyPolicy_for_CVPR_Challenge` in `src/mme_vla_suite/remote_evaluation/policy.py`).

### 2) Prepare the serving entrypoint

Copy and adapt the benchmark serving script into your repo:

- Reference: `https://github.com/RoboMME/robomme_benchmark/blob/main/scripts/challenge_serve_policy.py`
- In this repo, the entrypoint lives at `scripts/challenge_serve_policy.py`.

When you submit on EvalAI, provide a command like:

```bash
uv run ./scripts/challenge_serve_policy.py --checkpoint-dir my_cool_model_name/79999
```

### 3) Upload your checkpoint(s)

Upload your model checkpoint(s) somewhere the organizers can download them (e.g., Hugging Face).

- Example (replace with your own): `https://huggingface.co/<org_or_user>/<my_cool_model_name>`

### 4) Build the Docker image

From the repo root:

```bash
docker build -f docs/cvpr_challenge/Dockerfile -t my_cool_model_name:latest .
```

You may edit `docs/cvpr_challenge/Dockerfile` to include any additional dependencies your policy requires.

### 5) Push the Docker image to a registry

Push your image to a registry so the organizers/EvalAI can pull it (Docker Hub / GHCR / private registry).

Example (Docker Hub):

```bash
docker tag my_cool_model_name:latest <dockerhub_user>/my_cool_model_name:latest
docker login
docker push <dockerhub_user>/my_cool_model_name:latest
```

### 6) Submit on EvalAI

On EvalAI, provide:

- **Docker image** (registry path + tag), e.g. `<dockerhub_user>/my_cool_model_name:latest`
- **Command to start the policy server**, e.g.:

```bash
uv run ./scripts/challenge_serve_policy.py --host 0.0.0.0 --port 8001 --checkpoint-dir runs/ckpts/my_cool_model_name/79999
```

### Self-check before submission (recommended)

1) Run your container locally (maps the server port):

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -p 8001:8001 \
  my_cool_model_name:latest
```

2) Inside the container, start the policy server using your provided command.

3) From another terminal, run the benchmark eval client against your server:

- Reference eval script: `https://github.com/RoboMME/robomme_benchmark/blob/main/scripts/challenge_eval_policy.py`

---

### What the organizers will do

1) **Pull your image** (based on the image name/tag you provided in EvalAI), for example:

```bash
docker pull yinpeidai/my_cool_model_name:latest
```

2) **Download your checkpoint(s)** (based on the URL you provided), for example:

```bash
git clone https://huggingface.co/YinpeiDai/my_cool_model_name runs/ckpts/my_cool_model_name
```

3) **Run your container** (with a port mapping), for example:

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -p 8001:8001 \
  yinpeidai/my_cool_model_name:latest
```

Then, inside the container, start the policy server using the participant-provided command, for example:

```bash
uv run ./scripts/challenge_serve_policy.py --port 8001 --checkpoint-dir runs/ckpts/my_cool_model_name/79999
```

4) **Run evaluation** (phase 1), using the script from the RoboMME benchmark repo:

```bash
cd robomme_benchmark
uv run ./scripts/challenge_eval_policy.py --port 8001
```

After determining the top 5–10 teams, the organizers will run phase 2 evaluation.

