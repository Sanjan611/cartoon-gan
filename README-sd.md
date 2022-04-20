# AWS set up

- EC2 instance
    - Deep Learning Base AMI (Ubuntu 18)
    - Spot instance (cheaper), price limit $0.2
    - Configure security group (default)
    - (everything else, as is)
    - on EC2 dashboard, check the 'region' of the instance. Should be something like 'us-east-2c'.

- EBS volume
    - EC2 dashboard, click "Volumes" under EBS tab in the left pane.
    - Create volume
    - Availability zone: Make sure region of EBS is set to the same one as the instance
    - 100 GB
    - EBS volume should now be set up
    - Go back to "Volume" on EC2 dashboard
    - Right click on it, say "Attach volume"
    - Drop down instance and select the running one

- Terminal
    - ssh into the AWS instance
        - EC2 dashboard -> click on instance -> "Connect" (top right button) -> SSH -> copy the line at the bottom
    - run `lsblk`
        - should show the external ebs at the bottom of the list, like `nvme2n1`
    - under the default directory (which should be `/home/ubuntu/`), run - 
        - Format disk: `sudo mkfs -t ext4 /dev/nvme2n1`
        - Create mount directory: `mkdir data`
        - Mount: `sudo mount /dev/nvme2n1 data/`
        - Relax permissions: `sudo chmod go+rw data/`

All work like dataset, repo, etc. should be done inside this `data/` folder since that can be detached and attached to instances. Anything on instance storage will be deleted when instance gets terminated.
