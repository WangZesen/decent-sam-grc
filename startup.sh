# install UV and Python 3.10
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env
uv python install 3.10

# install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse -y

# mount GCS bucket
mkdir -p /root/gcs-bucket
gcsfuse --implicit-dirs my-training-log /root/gcs-bucket

# clone repo
cd /root
git clone https://github.com/WangZesen/decent-sam-grc.git
cd decent-sam-grc

# setup python environment
uv sync

# get instance name
INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)

# run
export XLA_IR_DEBUG=1
export TF_CPP_MIN_LOG_LEVEL=0
uv run image-cifar/test.py > /root/gcs-bucket/logs/${INSTANCE_NAME}_output.log 2>&1

# delete the mounted bucket
fusermount -u /root/gcs-bucket

# delete the queued resource
delete_qr() {
    local id=$1
    local zone="us-central2-b"
    local project="research-tpu-480810"
    
    echo "Attempting to delete Queued Resource: $id"

    gcloud alpha compute tpus queued-resources delete $id --zone=$zone --project=$project --quiet --force && return 0

    gcloud beta compute tpus queued-resources delete $id --zone=$zone --project=$project --quiet --force && return 0

    gcloud compute tpus queued-resources delete $id --zone=$zone --project=$project --quiet --force
}

QR_ID=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/queued-resource-id)
delete_qr $QR_ID
