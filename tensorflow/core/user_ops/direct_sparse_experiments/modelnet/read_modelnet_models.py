import numpy as np
import os
import random
import threading
import math

class ModelnetReader():
  def __init__(self, path, res, grid_size, batch_size, train=True, save = False, categories = None, preprocess = False, num_rotations = 30):
    self.path = path
    self.res = res
    self.grid_size = grid_size
    if categories == None:
      #self.categories = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
      self.categories = ["desk", "dresser"]
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
    self.thread = threading.Thread(target=self.run)
    self.preprocess = preprocess
    if self.preprocess:
      self.init()
      self.start()
      for sample in  self.samples:
        print("sample: ", sample[0])
        sample_indices = []
        sample_vals = []
        for i in range(0, num_rotations):
          angle = 2. * math.pi / num_rotations * i
          [inds, vals] = self.getData(self.path  + sample[0], angle)
          sample_indices.append(inds)
          sample_vals.append(vals)
          np.savetxt(self.path  + sample[0] + "_prep_val_" + str(i), vals, fmt='%f')
          np.savetxt(self.path  + sample[0] + "_prep_ind_" + str(i), inds, fmt='%i')

  
  def start(self):
    self.thread.start()
    return self.thread
  
  def getNumClasses(self):
    return len(self.categories)

  def init(self):
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
      [indices, values] = self.getData(self.path  + elem[0], i)
      self.batch[0].extend(indices)
      self.batch[1].extend(values)
      self.batch[3].append(elem[1])

  def run(self):
    self.load_batch()

  def next_batch(self):
    self.thread.join()
    data = self.batch
    has_data = False
    if len(self.samples) > self.batch_size:
      has_data = True
    self.thread = threading.Thread(target=self.run)
    return [data, has_data]
    
  def randomRotation2D(self, points, angle = None):
    if angle == None:
      angle = random.uniform(0, math.pi)
    for point in points:
      x = point[0]
      y = point[1]
      s = math.sin(angle)
      c = math.cos(angle)
      points[0] = x * c - y * s
      points[1] = y * c + x * s
    return points

  def getData(self, data_path, batch = 0, angle = None):
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
    for point in points:
      if point[0] > max_x:
        max_x = point[0]
      if point[1] > max_y:
        max_y = point[1]
      if point[2] > max_z:
        max_z = point[2]
      if point[0] < min_x:
        min_x = point[0]
      if point[1] < min_y:
        min_y = point[1]
      if point[2] < min_z:
        min_z = point[2]
    return [min_x, min_y, min_z, max_x, max_y, max_z]
