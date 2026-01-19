# install UV and Python 3.10
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.10

# install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse -y

# mount GCS bucket
mkdir -p $HOME/gcs-bucket
gcsfuse --implicit-dirs my-training-log $HOME/gcs-bucket

# clone repo
cd $HOME
git clone git@github.com:WangZesen/decent-sam-grc.git
cd decent-sam-grc

# setup python environment
uv sync

# run
uv run image-cifar/test.py

# touch a file to indicate the job is done
touch $HOME/gcs-bucket/DONE

# delete the mounted bucket
fusermount -u $HOME/gcs-bucket
rm -rf $HOME/gcs-bucket

# delete the queued resource
gcloud compute tpus queued-resources delete my-queue --zone=us-central2-b --quiet --force --project=$GRC_PROJECT