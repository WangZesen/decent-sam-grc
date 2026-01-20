import os
import time
import argparse
import datetime
import subprocess
from typing import Final
from loguru import logger

SOFTWARE_VERSION: Final[dict[str, str]] = {"v4": "tpu-ubuntu2204-base", "v5litepod": "v2-alpha-tpuv5-lite"}
TOPOLOGY_MAP: Final[dict[int, str]] = {
    8: "1x1x1",
    16: "2x1x1",
    32: "2x2x1",
    64: "2x2x2",
}
ALLOCATED_PAIRS = set([
    ("v4", "us-central2-b"),
    ("v5litepod", "europe-west4-b"),
    ("v5litepod", "us-central1-a"),
])

def fetch_queue_status(queue_name: str, zone: str) -> str:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        queue_name,
        f"--zone={zone}",
        '--format=value(state)',
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
        "--command={vm_command}"
    ]

    vm_command = "curl -LsSf https://astral.sh/uv/install.sh | sh\n" \
                 "source $HOME/.local/bin/env\n" \
                 "uv python install 3.10"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    out = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
    logger.info("Environment setup done")

    vm_command = 'export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`;' \
                 'echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list; ' \
                 'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -; ' \
                 'sudo apt-get update; ' \
                 'sudo apt-get install gcsfuse -y'
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    out = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
    logger.info("GCSFUSE installation done")

    vm_command = "mkdir -p $HOME/gcs-bucket; gcsfuse --implicit-dirs my-training-log $HOME/gcs-bucket"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    out = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
    logger.info("GCS bucket mount done")

    vm_command = "cd $HOME; rm -rf decent-sam-grc; git clone https://github.com/WangZesen/decent-sam-grc.git; cd decent-sam-grc; $HOME/.local/bin/uv sync"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    out = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
    logger.info("Repository clone done")

    vm_command = "cd $HOME/decent-sam-grc; " \
                 "export XLA_IR_DEBUG=1; " \
                 "export TF_CPP_MIN_LOG_LEVEL=0; " \
                 f"export timestamp={datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}; " \
                 "PJRT_DEVICE=TPU $HOME/.local/bin/uv run image-cifar/test.py > out.log 2>&1"
    command = [cmd.format(vm_command=vm_command) for cmd in vm_command_template]
    out = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
    logger.info(f"Training job output: {out}")



def run_job(args):
    try:
        assert (args.tpu, args.zone) in ALLOCATED_PAIRS, f"TPU type {args.tpu} is not available in zone {args.zone}."

        queue_name = create_tpu_vm_queue(tpu=args.tpu, num_cores=args.cores, zone=args.zone, time=args.duration)
        wait_for_queue_ready(queue_name, zone=args.zone)
        setup_env(queue_name, zone=args.zone)

        logger.info("TPU VM Queue is ready for use.")
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while creating or checking the TPU VM Queue: {e}")
    except AssertionError as ae:
        logger.error(str(ae))
    finally:
        # if 'queue_name' in locals():
        #     delete_tpu_vm_queue(queue_name, zone=args.zone)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage TPU VM Queues on GCP.")
    parser.add_argument("-z", "--zone", type=str, default="europe-west4-b", help="GCP zone for the TPU VM Queue.")
    parser.add_argument("-t", "--tpu", type=str, choices=["v4", "v5litepod"], default="v5litepod", help="Type of TPU.")
    parser.add_argument("-c", "--cores", type=int, default=8, help="Number of TPU cores.")
    parser.add_argument("-d", "--duration", type=str, default="8h", help="Duration for which the queue is valid.")
    args = parser.parse_args()

    run_job(args)
