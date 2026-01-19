#!/usr/bin/bash

# Create v6e-8 VM
gcloud compute tpus tpu-vm create test \
--accelerator-type=v6e-8 \
--version=v2-alpha-tpuv6e \
--zone=europe-west4-a \
--project=$GRC_PROJECT \
--spot