#!/usr/bin/bash

gcloud compute tpus queued-resources create my-queue \
    --zone=us-central2-b \
    --project=$GRC_PROJECT \
    --runtime-version=tpu-ubuntu2204-base \
    --accelerator-type=v4-8 \
    --node-count=2 \
    --subnetwork=my-tpu-subnet \
    --network=my-tpu-network \
    --metadata startup-script-url="gs://my-training-log/scripts/startup.sh" \
    --spot \
    --valid-until-duration=1h
