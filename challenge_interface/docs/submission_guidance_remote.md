# RoboMME Challenge Guide: Remote API Submission

This document explains how to serve your policy **from your own machine** (no Docker image submission) and submit a **public host + port** to EvalAI for CVPR challenge evaluation.

## What you (the participant) provide

- **A reachable policy server endpoint**: `<public_host_or_ip>:<port>`
- (Optional) **API key** for verification. We will include it in the [header](https://github.com/RoboMME/robomme_benchmark/challenge_interface/client.py#L35).

### 1) Implement the policy interface and serving script

Implement the `Policy` class compatible with the challenge [interface](https://github.com/RoboMME/robomme_benchmark/blob/edc8e8008718d9bf545cfcc2dd3dc2264c903239/src/remote_evaluation/policy.py#L23).

- Copy the [challenge_interface](https://github.com/RoboMME/robomme_benchmark/src/challenge_interface) directory from the benchmark repo into your repo.

  e.g., in this repo, we copied the participant-oriented files into the `challenge_interface` [directory](..).

  ```
  challenge_interface
  ├── __init__.py
  ├── msgpack_numpy.py
  ├── policy.py
  ├── scripts
  │   └── deploy.py
  └── server.py
  ```

- Override **`infer`** and **`reset`** in your policy implementation.
  
  e.g., we wrapped the original MME-VLA policy in the [`MyPolicy_for_CVPR_Challenge`](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/policy.py#L29) class for this challenge.

- Adjust `challenge_interface/scripts/deploy.py` for your own policy.

  e.g., in this repo, we modified [it](https://github.com/RoboMME/robomme_policy_learning/challenge_interface/scripts/deploy.py#L54) for the `MyPolicy_for_CVPR_Challenge` class.



### 2) Deploy your model

```
CUDA_VISIBLE_DEVICES=0  uv run python -m  challenge_interface.scripts.deploy --transport <type> --port <port> --checkpoint-dir xxx
```
You can deploy on your own server or a cloud machine, as long as it has a public IP.

### 3) Self-check locally with the benchmark eval client

Go to another machine, and run the [benchmark eval client](https://github.com/RoboMME/robomme_benchmark/challenge_interface/scripts/phase1_eval.py) to test your policy server:

```
cd <robomme_benchmark>
uv run python -m challenge_interface.scripts.phase1_eval --host <your_deployed_ip_or_dns>  --port <your_public_port> --transport <type>
```


### 4) Submit on EvalAI

On EvalAI, submit a JSON file that includes:

- **model_name**
- **email**
- **action_space**: you can only choose one from "joint_angle", "ee_pose", "waypoint".
- **evaluation_method**: set as `api`.
- **Host**
- **Port**
- **API key** (optional)
- **Transport**: Websocket or HTTP
- **Country/Area**: where the host machine is located. We will choose nodes that are close to your host machine to reduce latency.
- Other fields: `use_depth`, `use_camera_params` (default: `false`)

An example JSON file can be found [here](eval_ai_submission_example_remote.json).

---

## What the organizers will do

After we receive your submitted JSON file, we will

1) Connect to your submitted endpoint:

- `<your_public_host_or_ip>:<port>`

2) Run evaluation using the official benchmark eval script.

For example:
```bash
cd robomme_benchmark
uv run python -m  challenge_interface.scripts.phase1_eval --host <your_deployed_ip_or_dns> --port <your_public_port> --transport <type>
```

Keep your endpoint stable during the evaluation period.
