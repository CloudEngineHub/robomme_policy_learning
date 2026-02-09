# Manual evaluation (per model)

## π₀.₅ baseline
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/pi05_baseline/pi05_baseline/79999 --policy.config=pi05_baseline

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=pi05_baseline --args.model_ckpt_id=79999 --args.no-use-history
```


## MemER
MemER can be viewed as a combined use of symbolic and perceptual memory.

```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-memer 
```


## Symbolic MME-VLA

*SimpleSG + Oracle*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-oracle 
```
*SimpleSG + QwenVL*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-qwenvl 
```
*SimpleSG + Gemini*  
Set the `GOOGLE_API_KEY` environment variable when using Gemini.
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-gemini 
```


*GroundSG + Oracle*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-oracle 
```
*GroundSG + QwenVL*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-qwenvl 
```
*GroundSG + Gemini*  
Set the `GOOGLE_API_KEY` environment variable when using Gemini.
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-gemini 
```

## Perceptual MME-VLA

*TokenDrop + Context*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-context/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-context --args.model_ckpt_id=79999
```

*TokenDrop + Modulation*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-modul --args.model_ckpt_id=79999
```

*TokenDrop + Expert*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-expert --args.model_ckpt_id=79999
```


*FrameSamp + Context*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-context/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-context --args.model_ckpt_id=79999
```

*FrameSamp + Modulation*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-modul --args.model_ckpt_id=79999
```

*FrameSamp + Expert*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-expert --args.model_ckpt_id=79999
```


## Recurrent MME-VLA

*TTT + Context*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-context/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-context --args.model_ckpt_id=79999
```

*TTT + Modulation*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-modul --args.model_ckpt_id=79999
```

*TTT + Expert*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-expert --args.model_ckpt_id=79999
```


*RMT + Context*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-context/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-context --args.model_ckpt_id=79999
```

*RMT + Modulation*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-modul --args.model_ckpt_id=79999
```

*RMT + Expert*
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
source setup_robomme.bash
python examples/history_bench_sim/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-expert --args.model_ckpt_id=79999
```