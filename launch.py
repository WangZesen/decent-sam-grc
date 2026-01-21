import os
import time
import argparse
import datetime
import subprocess
from typing import Final
from loguru import logger

SOFTWARE_VERSION: Final[dict[str, str]] = {
    "v4": "tpu-ubuntu2204-base",
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}
TOPOLOGY_MAP: Final[dict[int, str]] = {
    8: "1x1x1",
    16: "2x1x1",
    32: "2x2x1",
    64: "2x2x2",
}
ALLOCATED_PAIRS = set(
    [
        ("v4", "us-central2-b"),
        ("v5litepod", "europe-west4-b"),
        ("v5litepod", "us-central1-a"),
        ("v6e", "europe-west4-a"),
    ]
)


def fetch_queue_status(queue_name: str, zone: str) -> str:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        queue_name,
        f"--zone={zone}",
        "--format=value(state)",
    ]
    result = subprocess.check_output(command)
    return result.decode("utf-8").strip().split("=")[-1]


def wait_for_queue_ready(queue_name: str, zone: str, poll_interval: int = 10) -> None:
    logger.info(f"Waiting for TPU VM Queue '{queue_name}' to be READY...")
    while True:
        status = fetch_queue_status(queue_name, zone)
        if status == "ACTIVE":
            logger.info(f"TPU VM Queue '{queue_name}' is READY.")
            break
        else:
            logger.info(f"Current status: {status}. Checking again in {poll_interval} seconds...")
            time.sleep(poll_interval)


def create_tpu_vm_queue(
    tpu: str,
    num_cores: int,
    zone: str,
    time: str = "72h",
    network: str = "tpu-vpc",
    subnetwork: str = "tpu-subnet",
) -> str:
    project = os.getenv("GRC_PROJECT")
    software_version = SOFTWARE_VERSION[tpu]
    queue_name = f"{tpu}-{num_cores}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    assert num_cores % 8 == 0, "Number of cores must be a multiple of 8."
    command = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        queue_name,
        f"--accelerator-type={tpu}-{num_cores}",
        f"--node-id={queue_name}",
        f"--runtime-version={software_version}",
        f"--zone={zone}",
        f"--project={project}",
        f"--network={network}",
        f"--subnetwork={subnetwork}",
        f"--valid-until-duration={time}",
        "--spot",
    ]
    subprocess.check_call(command)
    logger.info(f"Created TPU VM Queue: {queue_name} with {num_cores} cores in zone {zone}.")
    return queue_name


def delete_tpu_vm_queue(queue_name: str, zone: str) -> None:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        queue_name,
        f"--zone={zone}",
        "--quiet",
        "--force",
    ]
    subprocess.check_call(command)
    logger.info(f"Deleted TPU VM Queue: {queue_name}.")


def setup_env(queue_name: str, zone: str):
    vm_command_template = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        f"{queue_name}",
        f"--zone={zone}",
        "--worker=all",
        "--command={vm_command}",
    ]

    vm_command = "curl -LsSf https://astral.sh/uv/install.sh | sh\nsource $HOME/.local/bin/env\nuv python install 3.10"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    subprocess.check_call(command, stderr=subprocess.STDOUT)
    logger.info("Environment setup done")

    vm_command = "cd $HOME; rm -rf decent-sam-grc; git clone https://github.com/WangZesen/decent-sam-grc.git; cd decent-sam-grc; $HOME/.local/bin/uv sync"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    subprocess.check_call(command, stderr=subprocess.STDOUT)
    logger.info("Repository clone done")


def launch_jobs(queue_name: str, zone: str, job_list_dir: str, args):
    vm_command_template = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        f"{queue_name}",
        f"--zone={zone}",
        "--worker=all",
        "--command={vm_command}",
    ]
    check_finish_command_template = [
        "gsutil",
        "ls",
        "gs://my-training-log/{tag}_{index:04d}.done"
    ]

    configs = []
    tag = os.path.basename(job_list_dir).split("_")[-1].split(".")[0]
    with open(job_list_dir, "r") as f:
        line = f.readline()
        while line:
            line = line.strip()
            configs.append(line.split("\t")[1:])
            line = f.readline()
    index = 0

    while True:
        if index >= len(configs):
            logger.info("All jobs have been done!")
            break

        try:
            subprocess.check_call([cmd.format(tag=tag, index=index) for cmd in check_finish_command_template])
            logger.info(f"Job index {index:4d} is already completed. Skipping...")
            index += 1
            continue
        except subprocess.CalledProcessError:
            pass

        logger.info(f"Launching job index: {index}")
        configs_to_run = configs[index]

        vm_command = (
            "cd $HOME/decent-sam-grc/image-cifar; "
            f"PJRT_DEVICE=TPU $HOME/.local/bin/uv run -m src.decent_train {' '.join(configs_to_run)} > out.log 2>&1 && "
            f"gsutil mv out.log gs://my-training-log/{tag}_{index:04d}.log && "
            f"gsutil mv stats.csv gs://my-training-log/{tag}_{index:04d}.csv && "
            f"touch done && "
            f"gsutil mv done gs://my-training-log/{tag}_{index:04d}.done"
        )
        command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
        try:
            subprocess.check_call(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error(f"Job index {index} failed with error: {e}. Retrying...")
            status = fetch_queue_status(queue_name, zone)
            if status != "ACTIVE":
                logger.info(f"TPU VM Queue '{queue_name}' is not ACTIVE. Recreating...")
                delete_tpu_vm_queue(queue_name, zone)
                queue_name = create_tpu_vm_queue(tpu=args.tpu, num_cores=args.cores, zone=args.zone, time=args.duration)
                wait_for_queue_ready(queue_name, zone=args.zone)
                setup_env(queue_name, zone=args.zone)
                index -= 1  # Retry the same job
            else:
                raise Exception(f"TPU VM Queue is ACTIVE but job failed. Please check the logs. {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

        index += 1


def run_job(args):
    try:
        if args.queue_name:
            queue_name = args.queue_name
            logger.info(f"Using existing TPU VM Queue: {queue_name}")
        else:
            assert (args.tpu, args.zone) in ALLOCATED_PAIRS, f"TPU type {args.tpu} is not available in zone {args.zone}."
            queue_name = create_tpu_vm_queue(tpu=args.tpu, num_cores=args.cores, zone=args.zone, time=args.duration)
            wait_for_queue_ready(queue_name, zone=args.zone)
            setup_env(queue_name, zone=args.zone)
        launch_jobs(queue_name, zone=args.zone, job_list_dir=args.job_list, args=args)
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while creating or checking the TPU VM Queue: {e}")
    except AssertionError as ae:
        logger.error(f"Assertion Error: {ae}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # if 'queue_name' in locals():
        #     delete_tpu_vm_queue(queue_name, zone=args.zone)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage TPU VM Queues on GCP.")
    parser.add_argument("-z", "--zone", type=str, default="us-central1-a", help="GCP zone for the TPU VM Queue.")
    parser.add_argument(
        "-t", "--tpu", type=str, choices=["v4", "v5litepod", "v6e"], default="v5litepod", help="Type of TPU."
    )
    parser.add_argument("-c", "--cores", type=int, default=8, help="Number of TPU cores.")
    parser.add_argument("-d", "--duration", type=str, default="8h", help="Duration for which the queue is valid.")
    parser.add_argument("-q", "--queue-name", type=str, default="", help="Name of the TPU VM Queue to manage.")
    parser.add_argument("-l", "--job-list", type=str, help="Path to the job list file.", required=True)
    args = parser.parse_args()

    run_job(args)
