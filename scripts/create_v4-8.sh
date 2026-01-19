#!/usr/bin/bash

# Create v4-8 VM
gcloud compute tpus tpu-vm create test-v4-8 \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --zone=us-central2-b \
    --project=$GRC_PROJECT \
    --spot \
    --subnetwork=my-tpu-subnet \
    --network=my-tpu-network
