import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF
from snntorch import backprop
from snntorch import spikegen
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
import numpy as np
import cv2


#@title Plotting Settings
def plot_mem(mem, title=False):
  if title:
    plt.title(title)
  plt.plot(mem)
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, mem.shape[0]])
  plt.ylim([0, 1])
  plt.show()

def plot_step_current_response(cur_in, mem_rec, vline1):
  fig, ax = plt.subplots(2, figsize=(8,6),sharex=True)

  # Plot input current
  ax[0].plot(cur_in, c="tab:orange")
  ax[0].set_ylim([0, 0.2])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  ax[0].set_title("Lapicque's Neuron Model With Step Input")

  # Plot membrane potential
  ax[1].plot(mem_rec)
  ax[1].set_ylim([0, 0.6]) 
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

  if vline1:
    ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.xlabel("Time step")

  plt.show()


def plot_current_pulse_response(cur_in, mem_rec, title, vline1=False, vline2=False, ylim_max1=False):

  fig, ax = plt.subplots(2, figsize=(8,6),sharex=True)

  # Plot input current
  ax[0].plot(cur_in, c="tab:orange")
  if not ylim_max1:
    ax[0].set_ylim([0, 0.2])
  else:
    ax[0].set_ylim([0, ylim_max1])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem_rec)
  ax[1].set_ylim([0, 1])
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

  if vline1:
    ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  if vline2:
    ax[1].axvline(x=vline2, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.xlabel("Time step")

  plt.show()

def compare_plots(cur1, cur2, cur3, mem1, mem2, mem3, vline1, vline2, vline3, vline4, title):
  # Generate Plots
  fig, ax = plt.subplots(2, figsize=(8,6),sharex=True)

  # Plot input current
  ax[0].plot(cur1)
  ax[0].plot(cur2)
  ax[0].plot(cur3)
  ax[0].set_ylim([0, 0.2])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem1)
  ax[1].plot(mem2)
  ax[1].plot(mem3)
  ax[1].set_ylim([0, 1])
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

  ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  ax[1].axvline(x=vline2, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  ax[1].axvline(x=vline3, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  ax[1].axvline(x=vline4, ymin=0, ymax=2.2, alpha = 0.25, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)

  plt.xlabel("Time step")

  plt.show()

def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max2=1.25):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, 0.4])
  ax[0].set_xlim([0, 200])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  if title:
    ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2]) 
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  if thr_line:
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([]) 

  plt.show()

def plot_spk_mem_spk(spk_in, mem, spk_out, title):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [0.4, 1, 0.4]})

  # Plot input current
  splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_title(title)
  plt.yticks([]) 

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, 1])
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk_out, ax[2], s=400, c="black", marker="|")
  plt.ylabel("Output spikes")
  plt.yticks([]) 

  plt.show()


def plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0):
  # Generate Plots to Compare Reset Mechanisms
  fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [0.4, 1, 0.4], 'wspace':0.05})

  # Reset by Subtraction: input spikes
  splt.raster(spk_in, ax[0][0], s=400, c="black", marker="|")
  ax[0][0].set_ylabel("Input Spikes")
  ax[0][0].set_title("Reset by Subtraction")
  ax[0][0].set_yticks([])

  # Reset by Subtraction: membrane potential 
  ax[1][0].plot(mem_rec)
  ax[1][0].set_ylim([0, 0.7])
  ax[1][0].set_ylabel("Membrane Potential ($U_{mem}$)")
  ax[1][0].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)

  # Reset by Subtraction: output spikes
  splt.raster(spk_rec, ax[2][0], s=400, c="black", marker="|")
  ax[2][0].set_yticks([])
  ax[2][0].set_xlabel("Time step")
  ax[2][0].set_ylabel("Output Spikes")

  # Reset to Zero: input spikes
  splt.raster(spk_in, ax[0][1], s=400, c="black", marker="|")
  ax[0][1].set_title("Reset to Zero")
  ax[0][1].set_yticks([])

  # Reset to Zero: membrane potential
  ax[1][1].plot(mem_rec0)
  ax[1][1].set_ylim([0, 0.7])
  ax[1][1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  ax[1][1].set_yticks([])
  ax[2][1].set_xlabel("Time step")

  # Reset to Zero: output spikes
  splt.raster(spk_rec0, ax[2][1], s=400, c="black", marker="|")
  ax[2][1].set_yticks([])
  plt.show()


data_path = '../Spike-FlowNet/datasets/outdoor_day2/count_data_200fps/200.npy'
data = np.load(data_path)


conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias =False)
snn1 = snn.Leaky(beta = 0.99)
mem1 = snn1.init_leaky()
conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias =False)
snn2 = snn.Leaky(beta = 0.5, reset_mechanism="none")
mem2 = snn2.init_leaky()


mem1_rec = []
spk1_rec = []
mem2_rec = []
spk2_rec = []     
conv1.weight = torch.nn.Parameter(torch.ones_like(conv1.weight)*0.8) #manually set weights for input to first layer connection
conv2.weight = torch.nn.Parameter(torch.ones_like(conv2.weight)*0.15) #manually set weights for first to second layer connection
with torch.no_grad():
  conv2.weight[0,0,2,2] = 0.2

start_pos = 3000   # skip initial part of video
data_tensor = torch.tensor(data[0,:,:,start_pos:] + data[1,:,:,start_pos:])

print(conv2.weight)

with torch.no_grad():
  for step in range(int(data_tensor.size(2))):
    input = data_tensor[:,:,step].float()
    input = input[None,None,:]
    con_out = conv1(input)
    spk1, mem1 = snn1(con_out, mem1)
    con_out2 = conv2(spk1)
    spk2, mem2 = snn2(con_out2,mem2)
    img_lyr1 = torch.squeeze(spk1.detach())
    img_lyr1[img_lyr1>0] = 255 
    img_out = torch.squeeze(spk2.detach())
    img_out[img_out>0] = 255 
    img_in = torch.squeeze(input)
    img_in[img_in>0] = 255 
    img = np.hstack((img_in,img_out))
    cv2.imshow('Event Camera Output', np.array(img_in, dtype=np.uint8))
    cv2.waitKey(1)
    cv2.imshow('Network Output', np.array(img_out, dtype=np.uint8))
    cv2.waitKey(1)
    cv2.imshow('Layer1 Output', np.array(img_lyr1, dtype=np.uint8))
    cv2.waitKey(1)

    mem1_rec.append(torch.squeeze(mem1))
    spk1_rec.append(torch.squeeze(spk1))

    mem2_rec.append(torch.squeeze(mem2))
    spk2_rec.append(torch.squeeze(spk2))


# mem1_rec = torch.stack(mem1_rec)
# spk1_rec = torch.stack(spk1_rec)
# mem2_rec = torch.stack(mem2_rec)
# spk2_rec = torch.stack(spk2_rec)


# print(img.size())
# spike_in = data_tensor[0,100,100,:]
# spike_in = spk1_rec[:,100,100]
# spike_out = spk2_rec[:,100,100]
# mem_pot = mem2_rec[:,100,100]
# print(torch.sum(spike_in))
# print(torch.sum(spike_out))
# plot_mem(mem_pot)
# plot_spk_mem_spk(spike_in, mem_pot, spike_out, 'neuron activity')
# for i in range(data.shape[3]):
# cv2.imshow('Spike Image', np.array(data[1,:,:,i], dtype=np.uint8))
# cv2.waitKey(1)

