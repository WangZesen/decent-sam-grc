import hashlib
import argparse
import glob

parser = argparse.ArgumentParser(description="Generate config job list.")
parser.add_argument("-n", "--num-workers", type=int, default=8, help="Number of workers.")
args = parser.parse_args()


seed_configs = sorted(glob.glob("configs/seed/*.toml"))
p_configs = sorted(glob.glob("configs/mix/p/*.toml"))
start_configs = sorted(glob.glob("configs/mix/start/*.toml"))
topo_configs = sorted(glob.glob("configs/topo/*.toml"))
dataset_config = sorted(glob.glob("configs/dataset/*.toml"))
model_config = sorted(glob.glob("configs/model/*.toml"))
match args.num_workers:
    case 8:
        bs_configs = ["configs/bs/128.toml", "configs/bs/256.toml"]
    case 16:
        bs_configs = ["configs/bs/256.toml", "configs/bs/512.toml"]
    case 32:
        bs_configs = ["configs/bs/256.toml", "configs/bs/512.toml"]
    case _:
        raise ValueError(f"Unsupported number of workers: {args.num_workers}")

all_configs = []

for dataset in dataset_config:
    for model in model_config:
        for bs in bs_configs:
            for topo in topo_configs:
                for seed in seed_configs:
                    # decent baseline
                    all_configs.append([dataset, model, bs, topo, "configs/mix/normal.toml", seed])

                    for p in p_configs:
                        all_configs.append([dataset, model, bs, topo, p, "configs/mix/start/s=10.toml", seed])
                    for start in start_configs:
                        all_configs.append([dataset, model, bs, topo, "configs/mix/p/p=3.toml", start, seed])


# generate hash tag
concat_configs = seed_configs + p_configs + start_configs + topo_configs + dataset_config + model_config + bs_configs
tag = hashlib.md5(" ".join(concat_configs).encode("utf-8")).hexdigest()[:12]


# write to job list file
total = set()
index = 0

with open(f"jobs/job_list_{args.num_workers}_workers_{tag}.txt", "w") as f:
    for config in all_configs:
        config_line = "\t".join(config)
        if config_line in total:
            continue
        total.add(config_line)
        print(f"{index:04d}\t{config_line}", file=f)
        index += 1
