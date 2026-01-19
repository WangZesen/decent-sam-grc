gcloud compute tpus queued-resources create my-queue \
    --zone=us-central2-b \
    --project=$GRC_PROJECT \
    --version=tpu-ubuntu2204-base \
    --accelerator-type=v4-8 \
    --subnetwork=my-tpu-subnet \
    --network=my-tpu-network \
    --spot \
    --valid-until-duration=1h
