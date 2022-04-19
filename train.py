import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
import math
import numpy as np

import matplotlib.pyplot as plt

import os

from torch.utils.tensorboard import SummaryWriter

from generator import Generator, GeneratorLoss
from discriminator import Discriminator, DiscriminatorLoss
from torchvision import models
import torch.optim as optim
from os import listdir


image_size = 256
batch_size = 16

################# DATALOADER #######################

transformer = transforms.Compose([
	transforms.CenterCrop(image_size),
	transforms.ToTensor() # ToTensor() changes the range of the values from [0, 255] to [0.0, 1.0]
])

# Original cartoon images
cartoon_dataset = ImageFolder('../data/cartoons/', transformer)
len_training_set = math.floor(len(cartoon_dataset) * 0.9)
len_valid_set = len(cartoon_dataset) - len_training_set
training_set, _ = random_split(cartoon_dataset, (len_training_set, len_valid_set))
cartoon_image_dataloader_train = DataLoader(training_set, batch_size, shuffle=True, num_workers=0)

# Smoothed cartoon images
smoothed_cartoon_dataset = ImageFolder('../data/cartoons_smoothed/', transformer)
len_training_set = math.floor(len(smoothed_cartoon_dataset) * 0.9)
len_valid_set = len(smoothed_cartoon_dataset) - len_training_set
training_set, _ = random_split(smoothed_cartoon_dataset, (len_training_set, len_valid_set))
smoothed_cartoon_image_dataloader_train = DataLoader(training_set, batch_size, shuffle=True, num_workers=0)

# Real images
photo_dataset = ImageFolder('../data/photos/', transformer)
len_training_set = math.floor(len(photo_dataset) * 0.9)
len_valid_set = len(photo_dataset) - len_training_set
training_set, validation_set = random_split(photo_dataset, (len_training_set, len_valid_set))
photo_dataloader_train = DataLoader(training_set, batch_size, shuffle=True, num_workers=0)
photo_dataloader_valid = DataLoader(validation_set, batch_size, shuffle=True, num_workers=0)

################################ TENSORBOARD #####################
try:
	os.system("mkdir ../data/tensorboard/")
except:
	pass

tensorboard_logdir = '../data/tensorboard'
writer = SummaryWriter(tensorboard_logdir)

##################### MODELS ###################

G = Generator()
D = Discriminator()

############## DEVICE ##################

device = torch.device('cpu')

if torch.cuda.is_available():
	device = torch.device('cuda')
	print("Train on GPU.")
else:
	print("No cuda available")

G.to(device)
D.to(device)

############# VGG16 #######

path_to_pretrained_vgg16 = '../data/models/vgg16-397923af.pth'

vgg16 = models.vgg16(pretrained=True, progress = False) # Sanjan: couldn't get progress bar working
torch.save(vgg16, path_to_pretrained_vgg16)
vgg16 = vgg16.to(device) # Credits to bdlneto (https://github.com/bdlneto) https://github.com/TobiasSunderdiek/cartoon-gan/issues/5 for finding this issue and testing the new version

print(vgg16)

# due VGG16 has 5 pooling-layer, I assume conv4_4 is the 4th pooling layer
# (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
feature_extractor = vgg16.features[:24]
for param in feature_extractor.parameters():
	param.require_grad = False

print(feature_extractor)

################### LOSS FUNCTIONS ######################
discriminatorLoss = DiscriminatorLoss()
generatorLoss = GeneratorLoss(vgg16)

############# OPTIMIZER ###################
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

##################
os.system("mkdir ../data/models/checkpoints/")
os.system("mkdir -p ../data/intermediate_results/training/")
intermediate_results_training_path = "../data/intermediate_results/training/"

#################

def save_training_result(input, output):
	# input/output has batch-size number of images, get first one and detach from tensor
	image_input = input[0].detach().cpu().numpy()
	image_output = output[0].detach().cpu().numpy()
	# transponse image from torch.Size([3, 256, 256]) to (256, 256, 3)
	image_input = np.transpose(image_input, (1, 2, 0))
	image_output = np.transpose(image_output, (1, 2, 0))

	# generate filenames as timestamp, this orders the output by time
	filename = str(int(time.time()))
	path_input = intermediate_results_training_path + filename + "_input.jpg"
	path_output = intermediate_results_training_path + filename + ".jpg"
	plt.imsave(path_input, image_input)
	plt.imsave(path_output, image_output)


def write_model_weights_and_bias_to_tensorboard(prefix, state_dict, epoch):
	for param in state_dict:
		writer.add_histogram(f"{prefix}_{param}", state_dict[param], epoch)

import time
from tqdm import tqdm

def train(_num_epochs, checkpoint_dir, best_valid_loss, epochs_already_done, losses, validation_losses):
	init_epochs = 10
	print_every = 10
	start_time = time.time()

	for epoch in range(_num_epochs - epochs_already_done):
		epoch = epoch + epochs_already_done
		print(f"EPOCH:{epoch+1}/{_num_epochs}")

		for index, ((photo_images, _), (smoothed_cartoon_images, _), (cartoon_images, _)) in enumerate(zip(photo_dataloader_train, smoothed_cartoon_image_dataloader_train, cartoon_image_dataloader_train)):
			batch_size = photo_images.size(0)
			photo_images = photo_images.to(device)
			smoothed_cartoon_images = smoothed_cartoon_images.to(device)
			cartoon_images = cartoon_images.to(device)

			# train the discriminator
			d_optimizer.zero_grad()
			
			d_of_cartoon_input = D(cartoon_images)
			d_of_cartoon_smoothed_input = D(smoothed_cartoon_images)
			d_of_generated_image_input = D(G(photo_images))

			write_only_one_loss_from_epoch_not_every_batch_loss = (index == 0)

			d_loss = discriminatorLoss(d_of_cartoon_input,
										d_of_cartoon_smoothed_input,
										d_of_generated_image_input,
										epoch,
										write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss,
										writer = writer)

			d_loss.backward()
			d_optimizer.step()

			# train the generator
			g_optimizer.zero_grad()

			g_output = G(photo_images)

			d_of_generated_image_input = D(g_output)

			if epoch < init_epochs:
				# init
				init_phase = True
			else:
				# train
				init_phase = False

			g_loss = generatorLoss(d_of_generated_image_input,
									photo_images,
									g_output,
									epoch,
									is_init_phase=init_phase,
									write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss,
									writer = writer)

			g_loss.backward()
			g_optimizer.step()

			if (index % print_every) == 0:
				losses.append((d_loss.item(), g_loss.item()))
				now = time.time()
				current_run_time = now - start_time
				start_time = now
				print("Epoch {}/{} | d_loss {:6.4f} | g_loss {:6.4f} | time {:2.0f}s | total no. of losses {}".format(epoch+1, _num_epochs, d_loss.item(), g_loss.item(), current_run_time, len(losses)))

		# write to tensorboard
			#write_model_weights_and_bias_to_tensorboard('D', D.state_dict(), epoch)
			#write_model_weights_and_bias_to_tensorboard('G', G.state_dict(), epoch)
		# save some intermediate results during training
		print("Saving training results...")
		save_training_result(photo_images, g_output)

		# validate
		with torch.no_grad():
			D.eval()
			G.eval()

			for batch_index, (photo_images, _) in enumerate(photo_dataloader_valid):
				photo_images = photo_images.to(device)

				g_output = G(photo_images)
				d_of_generated_image_input = D(g_output)
				g_valid_loss = generatorLoss(d_of_generated_image_input,
												photo_images,
												g_output,
												epoch,
												is_init_phase=init_phase,
												write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss)

				if batch_index % print_every == 0:
					validation_losses.append(g_valid_loss.item())
					now = time.time()
					current_run_time = now - start_time
					start_time = now
					print("Epoch {}/{} | validation loss {:6.4f} | time {:2.0f}s | total no. of losses {}".format(epoch+1, _num_epochs, g_valid_loss.item(), current_run_time, len(validation_losses)))

		D.train()
		G.train()

		if(g_valid_loss.item() < best_valid_loss):
			print("Generator loss improved from {} to {}".format(best_valid_loss, g_valid_loss.item()))
			best_valid_loss = g_valid_loss.item()

		# save checkpoint
		checkpoint = {'g_valid_loss': g_valid_loss.item(),
						'best_valid_loss': best_valid_loss,
						'losses': losses,
						'validation_losses': validation_losses,
						'last_epoch': epoch+1,
						'd_state_dict': D.state_dict(),
						'g_state_dict': G.state_dict(),
						'd_optimizer_state_dict': d_optimizer.state_dict(),
						'g_optimizer_state_dict': g_optimizer.state_dict()
					}
		print("Save checkpoint for validation loss of {}".format(g_valid_loss.item()))
		torch.save(checkpoint, checkpoint_dir + '/checkpoint_epoch_{:03d}.pth'.format(epoch+1))
		if(best_valid_loss == g_valid_loss.item()):
			print("Overwrite best checkpoint")
			torch.save(checkpoint, checkpoint_dir + '/best_checkpoint.pth')

	return losses, validation_losses


checkpoint_dir = '../data/models/checkpoints'
checkpoints = listdir(checkpoint_dir)
num_epochs = 200 + 10 # training + init phase
epochs_already_done = 0
best_valid_loss = math.inf
losses = []
validation_losses = []

if(len(checkpoints) > 0):
	last_checkpoint = sorted(checkpoints)[-1]
	checkpoint = torch.load(checkpoint_dir + '/' + last_checkpoint, map_location=torch.device(device))
	best_valid_loss = checkpoint['best_valid_loss']
	epochs_already_done = checkpoint['last_epoch']
	losses = checkpoint['losses']
	validation_losses = checkpoint['validation_losses']

	D.load_state_dict(checkpoint['d_state_dict'])
	G.load_state_dict(checkpoint['g_state_dict'])
	d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
	g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
	print('Load checkpoint {} with g_valid_loss {}, best_valid_loss {}, {} epochs and total no of losses {}'.format(last_checkpoint, checkpoint['g_valid_loss'], best_valid_loss, epochs_already_done, len(losses)))


torch.cuda.empty_cache()

losses, validation_losses = train(num_epochs, checkpoint_dir, best_valid_loss, epochs_already_done, losses, validation_losses)