#!/usr/bin/bash

gcloud compute tpus queued-resources create on-demand-v4-32 \
    --zone=us-central2-b \
    --project=$GRC_PROJECT \
    --runtime-version=tpu-ubuntu2204-base \
    --accelerator-type=v4-8 \
    --node-count=4 \
    --subnetwork=my-tpu-subnet \
    --network=my-tpu-network \
    --valid-until-duration=168h
