import numpy as np
import os
import h5py
import argparse

parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS', help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='outdoor_day2', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='../datasets/outdoor_day2/outdoor_day2_data.hdf5', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()


save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
  os.makedirs(save_path)

count_dir = os.path.join(save_path, 'count_data_200fps')
if not os.path.exists(count_dir):
  os.makedirs(count_dir)
  
gray_dir = os.path.join(save_path, 'gray_data_200fps')
if not os.path.exists(gray_dir):
  os.makedirs(gray_dir)
  

class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)], shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0, gray=0, image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0, frame_rate=500):
        print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)
        print(input_event.shape)
 #       split_interval = image_raw_ts_temp.shape[0]
 #       data_split = 10 # N * (number of event frames from each groups)

        
        BIN_count = int(np.floor((image_raw_ts_temp[-1]-image_raw_ts_temp[0])*frame_rate))
        td_img_c = np.zeros((2, self.height, self.width, BIN_count), dtype=np.uint8)
        print(BIN_count)
        spike_timing = input_event[:,2]-input_event[0,2]
        last_element = 0
        
        i = 0
        new_element = 0 
        m = 0 
        frame_time = np.zeros((BIN_count,2))
        for i in range(BIN_count):
            print(i)
            frame_time[i,:] = [input_event[last_element,2],last_element]
            new_element = np.searchsorted(spike_timing[last_element:],(i+1)/frame_rate)
            frame_data = input_event[last_element:last_element+new_element,:]
            last_element += new_element

            print(frame_data.shape[0])
            for v in range(int(frame_data.shape[0])):
                if frame_data[v, 3].item() == -1:
                    td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), i] += 1
                elif frame_data[v, 3].item() == 1:
                    td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), i] += 1


        np.save(os.path.join(count_dir, str(frame_rate)), td_img_c)
        np.save(os.path.join(count_dir, "frame_time"), frame_time)

d_set = h5py.File(args.data_path, 'r')

raw_data = d_set['davis']['left']['events']
image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
gray_image = d_set['davis']['left']['image_raw']
d_set = None

dt_time = 1
fr_rate = 200; 

td = Events(raw_data.shape[0])
# Events
td.generate_fimage(input_event=raw_data, gray=gray_image, image_raw_event_inds_temp=image_raw_event_inds, image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time, frame_rate = fr_rate)
raw_data = None


print('Encoding complete!')
