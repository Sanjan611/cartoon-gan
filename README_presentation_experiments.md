# AWS instance set up

### EC2 instance
- Deep Learning Base AMI (Ubuntu 18)
- Spot instance (cheaper), price limit $0.2
- Select subnet as `us-east-2c` (doesn't matter, but just needs to be consistent)
- Configure security group (default)
- (everything else, as is)

### EBS volume
- EC2 dashboard, click "Volumes" under EBS tab in the left pane.
- Create volume
- Availability zone: Make sure region of EBS is set to the same one as the instance
- 100 GB
- EBS volume should now be set up
- Go back to "Volume" on EC2 dashboard
- Right click on it, say "Attach volume"
- Drop down instance and select the running one

### Terminal
- ssh into the AWS instance
    - EC2 dashboard -> click on instance -> "Connect" (top right button) -> SSH -> copy the line at the bottom
- run `lsblk`
    - should show the external ebs at the bottom of the list, like `nvme2n1`
- under the default directory (which should be `/home/ubuntu/`), run - 
    - Format disk: `sudo mkfs -t ext4 /dev/nvme2n1`. *NOTE*: This only needs to be done *once* the first time you create the volume. Doing it a second time will probably erase everything you stored.
    - Create mount directory: `mkdir data`
    - Mount: `sudo mount /dev/nvme2n1 data/`
    - Relax permissions: `sudo chmod go+rw data/`

All work like dataset, repo, etc. should be done inside this `data/` folder since that can be detached and attached to instances. Anything on instance storage will be deleted when instance gets terminated.

# Files and folders
Everything is now under the `data/` folder.

Clone the repo (main branch). For now, best switch to the branch I made, make a new branch off of that.

```
git clone https://github.com/Sanjan611/cartoon-gan.git
mkdir data
cd data
```

From our shared Drive folder, download the 3 *.zip* files (one for cartoon images, one for smoothed cartoon images, one for real photos). 

SCP those files to the `data/data/` directory on the instance. 

On the instance, first create folders like this under the `data/data/` folder.
```
cd ~/data/data/
mkdir cartoons
cd cartoons
mkdir 1

cd ..
mkdir cartoons_smoothed
cd cartoons_smoothed
mkdir 1

cd ..
mkdir photos
cd photos
mkdir 1

```

unzip those files like
```
cd ~/data/data
unzip safebooru.zip -d cartoons/1/
unzip safebooru-smoothed.zip -d cartoons_smoothed/1/
unzip coco.zip -d photos/1/
```

# Environment

### Python 3.8

```
cd ~/
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
```
Last command, just say `yes` to everything.

Then `exit` the instance and SSH back in to start conda environment `base` on start up.
### Libraries
```
cd ~/data/cartoon-gan
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install matplotlib
pip install tensorboard
```

# Experiment

Loss for the GAN training has two parts, the adversarial loss and the content loss. It's something like `L_adv + w * L_con` where w is like a scaling factor that controls how much content of original photo we want to preserve verses how cartoony stuff we want. 

The adversarial loss also contains terms that control the strength of edges (that are generally predominant in cartoon images). 

## Experiment 1: Varying w
The paper uses a w = 10 but the repo uses w = 5e-6. This is mainly cause the cartoon dataset used is different between the two and so some tuning is required. For this experiment, we'll run the GAN for w = 0 (no content preservation), w = 1e-10, w = 1e-5, w = 1. Hopefully we should see that there is a variation in the amount of cartoony stuff in the generated images.

This has been experimented with before in the repo.

In code, w is `content_loss_weight`.

- Shivani - w = 0 
- Suyash - w = 1e-10
- Sayali - w= 1e-5
- Sanjan - w = 1

In the `config_presentation.py` file, keep the other value `edge_loss_weight` = 1.


## Experiment 2: Adversarial loss terms
The paper says their loss terms are what sets them apart from other research in this area. So we'll just see how impactful the adversarial loss terms are to the output of the generator. 

One of the terms focuses on strengthening edges in the generated image. For this experiment we can vary the importance of that term and see what the generated output is like. This could be another scaling factor, something like `u * L_edge*.
We'll try with u = 0 (no edge considerations), u = 1e-5, u = 1, u = 10. 

This hasn't been experimented with before. So no idea how different the results are going to be.

In code, u is `edge_loss_weight`.

- Shivani - u = 0 
- Suyash - u = 1e-10
- Sayali - u= 1e-5
- Sanjan - u = 1

In the `config_presentation.py` file, keep the other value `content_loss_weight` = 5e-6.

# Steps to run

Depending on the experiment, change the value of `edge_loss_weight` and `content_loss_weight` in `config_presentation.py`.

Remember to use `tmux`. Create the tmux session with `tmux new -s train`. Then run `python train.py`. And hide the tmux session using `Ctrl + B` + `D`. It's a bit strange, like first press `Ctrl + B`, lift all your fingers and then just press `D`. To connect back to the tmux later on, do `tmux a -t train` (attaching).









