import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from data.willow_obj import WillowObject
from data.SPair71k import SPair71k
from utils.build_graphs import build_graphs

from utils.config import cfg
from torch_geometric.data import Data, Batch

datasets = {"PascalVOC": PascalVOC,
            "WillowObject": WillowObject,
            "SPair71k": SPair71k}

class GMDataset(Dataset):
    def __init__(self, name, length, **args):
        self.name = name
        self.ds = datasets[name](**args)
        self.true_epochs = length is None
        self.length = (
            self.ds.total_size if self.true_epochs else length
        )  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        if self.true_epochs:
            print(f"Initializing {self.ds.sets}-set with all {self.length} examples.")
        else:
            print(f"Initializing {self.ds.sets}-set. Randomly sampling {self.length} examples.")
        # length here represents the iterations between two checkpoints
        # if length is None the length is set to the size of the ds
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None
        self.num_graphs_in_matching_instance = None

    def set_cls(self, cls):
        if cls == "none":
            cls = None
        self.cls = cls
        if self.true_epochs:  # Update length of dataset for dataloader according to class
            self.length = self.ds.total_size if cls is None else self.ds.size_by_cls[cls]

    def set_num_graphs(self, num_graphs_in_matching_instance):
        self.num_graphs_in_matching_instance = num_graphs_in_matching_instance

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sampling_strategy = cfg.train_sampling if self.ds.sets == "train" else cfg.eval_sampling
        if self.num_graphs_in_matching_instance is None:
            raise ValueError("Num_graphs has to be set to an integer value.")

        idx = idx if self.true_epochs else None
        anno_list, perm_mat_list = self.ds.get_k_samples(idx, k=self.num_graphs_in_matching_instance, cls=self.cls, mode=sampling_strategy)
        for perm_mat in perm_mat_list:
            if (
                not perm_mat.size
                or (perm_mat.size < 2 * 2 and sampling_strategy == "intersection")
                and not self.true_epochs
            ):
                # 'and not self.true_epochs' because we assume all data is valid when sampling a true epoch
                next_idx = None if idx is None else idx + 1
                return self.__getitem__(next_idx)

        points_gt = [np.array([(kp["x"], kp["y"]) for kp in anno_dict["keypoints"]]) for anno_dict in anno_list]
        n_points_gt = [len(p_gt) for p_gt in points_gt]

        graph_list = []
        for p_gt, n_p_gt in zip(points_gt, n_points_gt):
            edge_indices, edge_features = build_graphs(p_gt, n_p_gt)

            # Add dummy node features so the __slices__ of them is saved when creating a batch
            pos = torch.tensor(p_gt).to(torch.float32) / 256.0
            assert (pos > -1e-5).all(), p_gt
            graph = Data(
                edge_attr=torch.tensor(edge_features).to(torch.float32),
                edge_index=torch.tensor(edge_indices, dtype=torch.long),
                x=pos,
                pos=pos,
            )
            graph.num_nodes = n_p_gt
            graph_list.append(graph)

        ret_dict = {
            "Ps": [torch.Tensor(x) for x in points_gt],
            "ns": [torch.tensor(x) for x in n_points_gt],
            "gt_perm_mat": perm_mat_list,
            "edges": graph_list,
        }

        imgs = [anno["image"] for anno in anno_list]
        if imgs[0] is not None:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
            imgs = [trans(img) for img in imgs]
            ret_dict["images"] = imgs
        elif "feat" in anno_list[0]["keypoints"][0]:
            feat_list = [np.stack([kp["feat"] for kp in anno_dict["keypoints"]], axis=-1) for anno_dict in anno_list]
            ret_dict["features"] = [torch.Tensor(x) for x in feat_list]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            raise ValueError("Cannot handle type {}".format(type(inp[0])))
        return ret

    ret = stack(data)
    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
    )
