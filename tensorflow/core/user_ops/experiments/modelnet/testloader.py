from read_modelnet_models import ModelnetReader
import time

reader = ModelnetReader("/home/thackel/Desktop/ModelNet10", 3, 8, 3, train=True, save=True)
reader.init()
reader.start()

for i in range(1, 100):
  t1 = time.time()
  batch = reader.next_batch()
  reader.start()
  t2 = time.time()
  print(batch)
  print("time: ", t2 - t1)
  print("start sleep")
  time.sleep(5)
  print("stop sleep")
