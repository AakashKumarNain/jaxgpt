Because the model was trained and tested on TPUv5p, the instructions here assume you are running this code on a TPU slice in a multi-host environment. If you are running on your own cluster, please change accordingly.

## 1. Setup

- Clone the repo to your local

```
https://github.com/AakashKumarNain/jaxgpt.git
```
- SCP the files to **all** the workers

```shell
gcloud alpha compute tpus tpu-vm scp --recurse jaxgpt ${TPU_NAME}: \
  --worker=all \
  --zone=${ZONE} \
  --project=${PROJECT_ID} 
```

- Create the python environment using `uv` as shown below:

```shell
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    cd ~/jaxgpt/
    uv venv --python=3.12
    source .venv/bin/activate
    uv sync
```
<br>

## 2. Cloud Storage and Dataset

We need storage for downloading the dataset, and saving the model checkpoints during the training. Though there are multiple [storage options](https://docs.cloud.google.com/tpu/docs/storage-options) available to work with, I prefer using buckets for short-training runs. Within buckets, you have the option to choose multi-regional, standard, or rapid storage. Here we will use the standard bucket, but in case you require lower latency, you can select other option too. For creating the bucket and providing the right permissions through IAM, please follow this [doc](https://docs.cloud.google.com/tpu/docs/storage-buckets#before_you_begin).

- We do not need to download and upload the dataset from all the workers. `ssh` into one of the workers

```shell
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0
```

- Activate the python environment we created above

```shell
cd jaxgpt
source .venv/bin/activate
```

- Download the tokenized dataset, and then upload it to the bucket for future use

```shell
python download_fineweb_tokens.py
gcloud storage rsync --recursive fineweb10B gs://YOUR_BUCKET_NAME/fineweb10B
```
<br>

## 3. Attaching bucket via FUSE

Once you create the bucket, you can mount the bucket at the path `/mnt/disks/data` (or any other path of your choice) using Cloud Storage [Fuse](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/overview) as show below:

```shell
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
    sudo apt-get update && sudo apt-get install -y gcsfuse
    sudo mkdir -p /mnt/disks/data
    sudo chown -R $USER:$USER /mnt/disks/data
    MY_UID=$(id -u)
    MY_GID=$(id -g)
    sudo mount -t gcsfuse -o rw,user,allow_other,uid=$MY_UID,gid=$MY_GID,implicit_dirs YOUR_BUCKET_NAME /mnt/disks/data/
  '
```
<br>

## 4. Train the model

Once we have successfully mounted our bucket, we can train the model as shown below:

```shell
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    cd ~/jaxgpt/
    uv venv --python=3.12
    source .venv/bin/activate
    uv sync
    cd ~/jaxgpt/gpt
    python -u train.py
  '
```

If you want to change any values in the architecture like #layers, #heads, etc, or any other hyperparameter, you can directly change it in the [config](../gpt/config.py) file. Once you make the changes, `scp` the changed files to all workers, similar to what is shown in point 2 except it won't be recursive this time.

<br>

## 5. Inference

Similar to training, you can run the inference by running the following:

```shell
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
  cd ~/jaxgpt/
  source .venv/bin/activate
  cd ~/jaxgpt/gpt
  python -u inference.py
'
```

