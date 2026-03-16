from mme_vla_suite.remote_evaluation.server import PolicyServer
from mme_vla_suite.policies.policy_config import create_trained_policy
from mme_vla_suite.remote_evaluation.policy import MyPolicy_for_CVPR_Challenge
from pathlib import Path
from mme_vla_suite.training.config import get_config



###### Participant Parameters ######
# You will submit a json file including the following parameters at eval.ai 
HOST = "141.212.115.116"
PORT = 8012
####################################


model = create_trained_policy(
    train_config=get_config("mme_vla_suite"),
    checkpoint_dir=Path("runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999"),
    seed=7,
)

policy = MyPolicy_for_CVPR_Challenge(model=model)
server = PolicyServer(policy, host=HOST, port=PORT)
server.serve_forever()