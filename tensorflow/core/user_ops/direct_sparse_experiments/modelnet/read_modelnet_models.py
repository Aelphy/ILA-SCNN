import numpy as np
import os
import random
import math

class ModelnetReader():
  def __init__(self, path, res, grid_size, batch_size, train=True, save = False, categories = None, preprocess = False, num_rotations = 30):
    self.path = path
    self.res = res
    self.grid_size = grid_size
    if categories == None:
      self.categories = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
      #self.categories = ["desk", "dresser"]
      #self.categories = ["tdesk", "tdresser"]
    else:
      self.categories = categories
    self.train_dir = "/train"
    self.test_dir = "/test"
    self.train = train
    self.batch_size = batch_size
    self.samples = []
    self.batch = []
    self.save = save
    random.seed()
    self.preprocess = preprocess
    self.preloaded_samples = []
    self.preloaded_labels = []
    if self.preprocess:
      for i in range(0, len(self.categories)):
        class_label = [0] * len(self.categories)
        class_label[i] = 1
        if self.train:
          append = "/" + self.categories[i] + self.train_dir
        else:
          append = "/" + self.categories[i] + self.test_dir
        path_ = self.path + append
        for file in os.listdir(path_):
          if file.endswith("xyz"):
            data_path = path_ + "/" + file
            points = np.loadtxt(data_path)
            self.preloaded_samples.append(points)
            self.preloaded_labels.append(class_label)
  
  def start(self):
    return

  def getNumClasses(self):
    return len(self.categories)

  def init(self):
    if self.preprocess:
      self.samples = range(0, len(self.preloaded_samples))
      random.shuffle(self.samples)
      return
    self.samples = []
    for i in range(0, len(self.categories)):
      class_label = [0] * len(self.categories)
      class_label[i] = 1
      if self.train:
        append = "/" + self.categories[i] + self.train_dir
      else:
        append = "/" + self.categories[i] + self.test_dir
      path_ = self.path + append
      for file in os.listdir(path_):
        if file.endswith("xyz"):
          self.samples.append((append + "/" + file, class_label))
          #self.getData(path_ + "/" + file)
    if self.train:
      random.shuffle(self.samples)

  def load_batch(self):
    shape = [self.batch_size, self.res, self.res, self.res, 1]
    self.batch = [[],[],shape,[]]
    if len(self.samples) < self.batch_size:
      return
    for i in range(0, self.batch_size):
      elem = self.samples.pop()
      if self.preprocess:
        [indices, values] = self.getData(elem, i)
        label = self.preloaded_labels[elem]
      else:
        [indices, values] = self.getData(self.path  + elem[0], i)
        label = elem[1]
      self.batch[0].extend(indices)
      self.batch[1].extend(values)
      self.batch[3].append(label)


  def next_batch(self):
    if len(self.samples) > self.batch_size:
      self.load_batch()
      has_data = True
    else:
      has_data = False
    data = self.batch
    return [data, has_data]
    
  def randomRotation2D(self, points, angle = None):
    if angle == None:
      angle = random.uniform(0, math.pi)
   
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = x * np.cos(angle) - y * np.sin(angle)
    points[:, 1] = y * np.cos(angle) + x * np.sin(angle)
    return points

  def getData(self, data_path, batch = 0, angle = None):
    if self.preprocess:
      points = self.preloaded_samples[data_path]
    else:
      points = np.loadtxt(data_path)

    if self.train:
      points = self.randomRotation2D(points, angle)
    [indices, values] = self.toVoxelGrid(points, batch)
    return [indices, values]

  def toVoxelGrid(self, points, batch):
    [min_x, min_y, min_z, max_x, max_y, max_z] = self.getBoundingBox(points)
    off_x = float(max_x - min_x) / 2 - min_x
    off_y = float(max_y - min_y) / 2 - min_y
    off_z = float(max_z - min_z) / 2 - min_z
    max_size = max([max_x + off_x, max_y + off_y, max_z + off_z, self.grid_size])
    scale = float(max_size) / self.res + 1 
    sparse_grid = {}
      
    for point in points:
      bin_x = int((point[0] + off_x) / scale)
      bin_y = int((point[1] + off_y) / scale)
      bin_z = int((point[2] + off_z) / scale)
      pid = (int(batch), bin_x, bin_y, bin_z, 0)
      sparse_grid[pid] = float(1)
    sparse_points = np.array(sparse_grid.keys(), dtype=np.int64)
    #print("sparse points: ", sparse_points)
    sparse_values = sparse_grid.values()
    if self.save:
      np.savetxt("grid_points.xyz", sparse_points, fmt='%i')
    return [sparse_points.tolist(), sparse_values]

  def getBoundingBox(self, points):
    min_x = 100000
    max_x = -100000
    min_y = 100000
    max_y = -100000
    min_z = 100000
    max_z = -100000
    
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()
    max_z = points[:, 2].max()
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    min_z = points[:, 2].min()

    return [min_x, min_y, min_z, max_x, max_y, max_z]
