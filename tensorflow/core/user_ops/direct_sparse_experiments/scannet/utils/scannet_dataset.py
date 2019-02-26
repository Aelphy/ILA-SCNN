import os
import cPickle as pickle
import numpy as np
import os.path as osp
import open3d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tqdm import tqdm


class ScanNetDataset():
    def __init__(self, path, res, split=False, save_preloads='../data'):
        self.data_locs = {}
        self.data = {}
        self.save_preloads = save_preloads
        self.split = split
        self.path = path
        self.res = res
        self.labels = []
        self.locs = []
        self.train_labels = None
        self.val_labels = None
        self.train_locs = None
        self.val_locs = None
        self.train_queue = []
        self.val_queue = []
    
        for scene in os.listdir(path):
            with open(osp.join(path, scene, scene + '.txt')) as f:
                for line in f:
                    if 'sceneType' in line:
                        label = line.split('=')[1].strip()
                        break
                        
            if label not in self.data_locs:
                self.data_locs[label] = []
                
            self.data_locs[label].append(osp.join(scene, scene + '_vh_clean.ply'))
        
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit_transform(sorted(self.data_locs.keys()))
        
        for k, v in self.data_locs.items():
            for e in v:
                self.labels.append(self.label_encoder.transform([k])[0])
                self.locs.append(e)
        
        self.labels = np.array(self.labels)
        if split:
            self.train_locs = []
            self.val_locs = []
            self.train_labels = []
            self.val_labels = []
            
            for k, v in self.data_locs.items():
                train_idxs, val_idxs = train_test_split(np.arange(len(v)), random_state=42)
                self.train_locs += [v[i] for i in train_idxs]
                self.train_labels.append(self.label_encoder.transform([k])[0] * np.ones(len(train_idxs)))
                self.val_locs += [v[i] for i in val_idxs]
                self.val_labels.append(self.label_encoder.transform([k])[0] * np.ones(len(val_idxs)))
                
            self.train_labels = np.concatenate(self.train_labels)
            self.val_labels = np.concatenate(self.val_labels)
            
        if save_preloads is not None:
            if osp.exists(osp.join(save_preloads, 'data.pkl')):
                with open(osp.join(save_preloads, 'data.pkl')) as f:
                    self.data = pickle.load(f)
            else:
                for elem in tqdm(self.locs):
                    self.data[elem] = self.get_data(osp.join(self.path, elem)).astype(np.float16)
                    
                
                with open(osp.join(save_preloads, 'data.pkl'), 'wb') as f:
                    pickle.dump(self.data, f)


    def get_num_classes(self):
        return len(self.data_locs)

    def iterate_minibatches(self, batch_size, part=None, shuffle=True, angle=-1):
        shape = [batch_size, self.res, self.res, self.res, 1]
        if self.split:
            if part == 'train':
                db = self.train_locs
                ls = self.train_labels
                queue = self.train_queue
            elif part == 'val':
                db = self.val_locs
                ls = self.val_labels
                queue = self.val_queue
            else:
                raise RuntimeError('wrong part, should be train or val')
        else:
            db = self.locs
            ls = self.labels
            # TODO: fix queue behaviour
                
        if len(queue) < 10 * batch_size:
            if shuffle:
                classes = np.arange(self.get_num_classes())
                np.random.shuffle(classes)
                for i in classes:
                    queue.append((np.array(db)[ls == i][np.random.choice(len(ls[ls == i]))], i))
            else:
                for i in range(0, len(db) - batch_size + 1, batch_size):
                    for j, elem in enumerate(db[i:i + batch_size]):
                        queue.append((elem, ls[j + i]))
        
        if part == 'val':
            for i in range(0, len(queue) - batch_size + 1, batch_size):
                batch = [[], [], shape, []]
                for j in range(i, i + batch_size):
                    elem, cl = queue.pop()

                    if self.save_preloads is not None:
                        points = self.data[elem]
                    else:
                        points = self.get_data(osp.join(self.path, elem))

                    indicies, values = self.preprocess(points, angle)
                    batch[0].extend(np.concatenate([j * np.ones((len(indicies), 1)).astype(np.int64), indicies], axis=1))
                    batch[1].extend(values)
                    batch[3].append(cl)

                batch[3] = to_categorical(batch[3], num_classes=self.get_num_classes())
                yield batch
        else:
            batch = [[], [], shape, []]
            for i in range(batch_size):
                elem, cl = queue.pop()

                if self.save_preloads is not None:
                    points = self.data[elem]
                else:
                    points = self.get_data(osp.join(self.path, elem))

                indicies, values = self.preprocess(points, angle)
                batch[0].extend(np.concatenate([i * np.ones((len(indicies), 1)).astype(np.int64), indicies], axis=1))
                batch[1].extend(values)
                batch[3].append(cl)

            batch[3] = to_categorical(batch[3], num_classes=self.get_num_classes())
            yield batch

    
    def random_rotation_2D(self, points, angle=-1):
        if angle == -1:
            angle = np.random.uniform(0, np.pi)

        x = points[:, 0].copy()
        y = points[:, 1].copy()
        result = points.copy()
        result[:, 0] = x * np.cos(angle) - y * np.sin(angle)
        result[:, 1] = y * np.cos(angle) + x * np.sin(angle)
        return result

    
    def get_data(self, data_path):        
        return np.asarray(open3d.read_point_cloud(data_path).points)
    
    
    def preprocess(self, points, angle):
        if angle is not None:
            points = self.random_rotation_2D(points, angle)
        
        return self.to_voxel_grid(points)

    
    def to_voxel_grid(self, points):
        # Adapted from pyntcloud
        x_y_z = [self.res, self.res, self.res]

        xyzmin = points.min(0)
        xyzmax = points.max(0)

        segments = []
        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)

        # find where each point lies in corresponding segmented axis
        # -1 so index are 0-based; clip for edge cases
        voxel_x = np.clip(np.searchsorted(segments[0], points[:, 0]) - 1, 0, x_y_z[0])
        voxel_y = np.clip(np.searchsorted(segments[1], points[:, 1]) - 1, 0, x_y_z[1])
        voxel_z = np.clip(np.searchsorted(segments[2], points[:, 2]) - 1, 0,  x_y_z[2])

        return np.concatenate((np.vstack([voxel_x, voxel_y, voxel_z]).T, np.zeros((len(voxel_x), 1))), axis=-1).astype(np.int64), 0.1 * np.ones(len(voxel_x))