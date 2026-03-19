## RoboMME Challenge Guide: Docker Submission 

This document explains how to package your policy into a Docker image that the organizers can pull and run for CVPR challenge evaluation.

We use a MME-VLA (framesamp+modul) as an example to walk you through.

### What you (participant) provide

- **A Docker image** containing your policy server code and all dependencies.
- **A checkpoint location** that the organizers can download  
e.g., a Hugging Face repo https://huggingface.co/Yinpei/perceptual-framesamp-modul 

- **One command** to start your policy server inside the container.  
e.g. `uv run python -m  challenge_inteface.scripts.deploy --port 8001 --checkpoint-dir perceptual-framesamp-modul/79999`

### 1) Implement the policy interface and serving script

Implement the `Policy` class compatible with the challenge [interface](https://github.com/RoboMME/robomme_benchmark/blob/edc8e8008718d9bf545cfcc2dd3dc2264c903239/src/remote_evaluation/policy.py#L23).

- Copy the [`challenge_inteface/` directory](`https://github.com/RoboMME/robomme_benchmark/tree/main/src/challenge_inteface) from benchamrk repo into your repo. 

  e.g. we copy the participant-used files to `src/mme_vla_suite/challenge_inteface` in this repo.

- Override **`infer`** and **`reset`** in your policy implementation 
  
   e.g, we wrap up the original MME-VLA policy into [`MyPolicy_for_CVPR_Challenge](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/policy.py#L29) class for the challenge.

- Adjust `challenge_interface/scripts/deploy.py` for you own pllicy.

  e.g.,  In this repo, we modify it for `MyPolicy_for_CVPR_Challenge` class


### 2) Upload your checkpoint(s)

Upload your model checkpoint(s) somewhere the organizers can download them (e.g., Hugging Face).

- Example (replace with your own): `https://huggingface.co/<org_or_user>/<my_cool_model_name>`

### 4) Build the Docker image

We provide a `docs/cvpr_challenge/Dockerfile` example.

You may edit to include any additional dependencies your policy requires or use your own dockerfile.

Build the docker image

```bash
docker build -f docs/cvpr_challenge/Dockerfile -t <my_cool_model_name>:latest .
```

### 5) Push the Docker image to a registry

Push your image to a registry so the organizers can pull it from Docker Hub.

```bash
docker tag <my_cool_model_name>:latest <dockerhub_user>/<my_cool_model_name>:latest
docker login
docker push <dockerhub_user>/<my_cool_model_name>:latest
```

For eaxmple, the organizers push a image for [framesamp+modul](https://hub.docker.com/repository/docker/yinpeidai/my_cool_model_name/general) to docker hub.

### 6) Submit on EvalAI

On EvalAI, submit a json file include:

- ...
- **Docker image** (registry path + tag), e.g. `<dockerhub_user>/my_cool_model_name:latest`
- **Command to start the policy server**, e.g.:
- ...

An example json file can be found [here](eval_ai_submission_example.json).


### Self-check before submission (recommended)

1) Run your container locally (maps the server port):

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -p 8001:8001 \
  my_cool_model_name:latest
```

2) Inside the container, start the policy server using your modified `deploy.py`
```
uv run python -m  challenge_interface.scripts.deploy ...
```

3) From another terminal, run the benchmark eval client against your server:

```
cd robomme_benchmark
uv run python -m  challenge_inteface.scripts.phase1_eval
```

---

### What the organizers will do

1) **Pull your image** (based on the image name/tag you provided in EvalAI), for example:

```bash
docker pull yinpeidai/my_cool_model_name:latest
```

2) **Download your checkpoint(s)** (based on the URL you provided), for example:

```bash
git clone https://huggingface.co/YinpeiDai/perceptual-framesamp-modul runs/ckpts/perceptual-framesamp-modul
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
uv run python -m  challenge_interface.scripts.deploy --port 8001 --checkpoint-dir runs/ckpts/perceptual-framesamp-modul/79999
```

4) **Run evaluation** (phase 1), using the script from the RoboMME benchmark repo:

```bash
cd robomme_benchmark
uv run ./scripts/challenge_eval_policy.py --port 8001
```

After determining the top 5–10 teams, the organizers will run phase 2 evaluation.

