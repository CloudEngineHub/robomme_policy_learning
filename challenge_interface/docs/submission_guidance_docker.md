# RoboMME Challenge Guide: Docker Submission

This document explains how to package your policy into a Docker image that organizers can pull and run for CVPR challenge evaluation.

We use an MME-VLA (framesamp+modul) model as an example.

## What you (the participant) provide

- **A Docker image** containing your policy server code and all dependencies.
- **A checkpoint location** that the organizers can download.
- **One command** to start your policy server inside the container.  

### 1) Implement the policy interface and serving script

Implement the `Policy` class compatible with the challenge [interface](https://github.com/RoboMME/robomme_benchmark/blob/edc8e8008718d9bf545cfcc2dd3dc2264c903239/src/remote_evaluation/policy.py#L23).

- Copy the [challenge_inteface](https://github.com/RoboMME/robomme_benchmark/src/challenge_inteface) directory from the benchmark repo into your repo.

  e.g., in this repo, we copied the participant-facing files into the `challenge_inteface` [directory](..).

- Override **`infer`** and **`reset`** in your policy implementation.
  
  e.g., we wrapped the original MME-VLA policy in the [`MyPolicy_for_CVPR_Challenge`](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/policy.py#L29) class for this challenge.

- Adjust `challenge_interface/scripts/deploy.py` for your own policy.

  e.g., in this repo, we modified it for the `MyPolicy_for_CVPR_Challenge` class.


### 2) Upload your checkpoint(s)

Upload your model checkpoint(s) somewhere the organizers can download them.

- For example, we uploaded the framesamp+modul MME-VLA model to `https://huggingface.co/Yinpei/perceptual-framesamp-modul`.

### 4) Build the Docker image

We provide a `challenge_interface/docs/Dockerfile` example.

You may edit it to include any additional dependencies your policy requires, or use your own Dockerfile.

Build the Docker image:

```bash
docker build -f challenge_interface/docs/Dockerfile -t <my_cool_model_name>:latest .
```


### 5) Self-check locally with the benchmark eval client

1) Run your container locally (maps the server port):

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -p 8001:8001 \
  my_cool_model_name:latest
```
We put all the model ckpts under the `runs` directory.

2) Inside the container, start the policy server using your modified `deploy.py`.

3) From another terminal, run the [benchmark eval client](https://github.com/RoboMME/robomme_benchmark/challenge_inteface/scripts/phase1_eval.py) against your server:

```
cd robomme_benchmark
uv run python -m  challenge_inteface.scripts.phase1_eval --port 8001
```

### 6) Push the Docker image to a registry

Push your image to a registry so the organizers can pull it from Docker Hub.

```bash
docker tag <my_cool_model_name>:latest <dockerhub_user>/<my_cool_model_name>:latest
docker login
docker push <dockerhub_user>/<my_cool_model_name>:latest
```

For example, organizers pushed an image for [framesamp+modul](https://hub.docker.com/repository/docker/yinpeidai/my_cool_model_name/general) to Docker Hub.


### 7) Submit on EvalAI

On EvalAI, submit a JSON file that includes:

- **model_name**
- **email**
- **action_space**: you can only choose one from "joint_angle", "ee_pose", "waypoint".
- **evaluation_method**: set as `docker`.
- **Checkpoint URL** (downloadable by organizers), e.g. `https://huggingface.co/Yinpei/perceptual-framesamp-modul`
- **Docker image** (registry path + tag), e.g. `<dockerhub_user>/my_cool_model_name:latest`
- **Command to start the policy server**, e.g. `uv run python -m challenge_interface.scripts.deploy --checkpoint-dir runs/ckpts/perceptual-framesamp-modul/79999`. The organizers will run `deploy.py` to start your policy server, then run evaluation.
- others: `use_depth`, `use_camera_params` (default false)

An example JSON file can be found [here](eval_ai_submission_example_docker.json).


---

## What the organizers will do

1) **Pull your image** (based on the image name/tag you provided in EvalAI), for example:

```bash
docker pull yinpeidai/perceptual-framesamp-modul:latest
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
  yinpeidai/perceptual-framesamp-modul:latest
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

