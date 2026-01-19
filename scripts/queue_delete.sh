#!/usr/bin/bash

gcloud compute tpus queued-resources delete my-queue --zone=us-central2-b --quiet --force --project=$GRC_PROJECT