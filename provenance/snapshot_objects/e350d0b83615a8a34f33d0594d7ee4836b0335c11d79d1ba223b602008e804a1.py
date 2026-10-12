import hashlib
import logging
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from data.data import iCIFAR224, iImageNetR, iImageNetA, CUB, vtab, iCARS196_224, omnibenchmark, objectnet


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args or {}
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed, self.args)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        expected_tasks = self.args.get("expected_tasks")
        if expected_tasks is not None and self.nb_tasks != int(expected_tasks):
            raise ValueError(f"expected {expected_tasks} tasks, constructed {self.nb_tasks}: {self._increments}")
        self.protocol_manifest = self._build_protocol_manifest()

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_class_metadata(self, mapped_class_id):
        mapped_class_id = int(mapped_class_id)
        original_class_id = int(self._class_order[mapped_class_id])
        class_name = None
        if self.class_names and original_class_id < len(self.class_names):
            class_name = self.class_names[original_class_id]
        return {
            "mapped_class_id": mapped_class_id,
            "original_class_id": original_class_id,
            "class_name": class_name,
        }

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, args):
        idata = _get_idata(dataset_name)
        idata.download_data(args)
        # train_data和train_target和test_data和test_target

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self.class_names = getattr(idata, "class_names", [str(i) for i in sorted(np.unique(self._train_targets))])
        self.split_roots = getattr(idata, "split_roots", {})
        expected_classes = args.get("expected_num_classes")
        actual_classes = len(np.unique(self._train_targets))
        if expected_classes is not None and actual_classes != int(expected_classes):
            raise ValueError(f"expected {expected_classes} classes, found {actual_classes}")
        if set(np.unique(self._train_targets)) != set(np.unique(self._test_targets)):
            raise ValueError("train/test class label sets differ")

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    @staticmethod
    def _split_digest(root, verify_content=True):
        root = Path(root)
        digest = hashlib.sha256()
        relative_paths = []
        basenames = []
        content_hashes = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            if verify_content:
                file_digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                content_hash = file_digest.hexdigest()
                digest.update(b"\0")
                digest.update(content_hash.encode("ascii"))
                content_hashes.append(content_hash)
            digest.update(b"\n")
            relative_paths.append(relative)
            basenames.append(path.name)
        return {
            "root": str(root.resolve()),
            "file_count": len(relative_paths),
            "sha256": digest.hexdigest(),
            "hash_mode": (
                "sorted_relative_path_file_size_and_content_sha256"
                if verify_content
                else "sorted_relative_path_and_file_size"
            ),
            "relative_paths": set(relative_paths),
            "basenames": set(basenames),
            "content_hashes": set(content_hashes),
        }

    def _build_protocol_manifest(self):
        tasks = []
        start = 0
        for task_id, size in enumerate(self._increments):
            mapped_ids = list(range(start, start + size))
            original_ids = self._class_order[start : start + size]
            tasks.append(
                {
                    "task_id": task_id,
                    "mapped_class_ids": mapped_ids,
                    "original_class_ids": original_ids,
                    "class_names": [self.class_names[index] for index in original_ids],
                }
            )
            start += size
        manifest = {
            "dataset": self.dataset_name,
            "num_classes": len(self._class_order),
            "num_tasks": len(self._increments),
            "increments": list(self._increments),
            "class_order_original_ids": list(self._class_order),
            "tasks": tasks,
            "label_remapping": "new_label = index of original ImageFolder label in class_order_original_ids",
            "train_samples_per_mapped_class": dict(sorted(Counter(map(int, self._train_targets)).items())),
            "test_samples_per_mapped_class": dict(sorted(Counter(map(int, self._test_targets)).items())),
        }
        if self.split_roots:
            verify_content = bool(self.args.get("verify_split_content", True))
            train = self._split_digest(self.split_roots["train"], verify_content)
            test = self._split_digest(self.split_roots["test"], verify_content)
            manifest["splits"] = {
                "train": {key: value for key, value in train.items() if not isinstance(value, set)},
                "test": {key: value for key, value in test.items() if not isinstance(value, set)},
                "overlap": {
                    "relative_path_count": len(train["relative_paths"] & test["relative_paths"]),
                    "basename_count": len(train["basenames"] & test["basenames"]),
                    "content_sha256_count": (
                        len(train["content_hashes"] & test["content_hashes"])
                        if verify_content
                        else None
                    ),
                    "note": "content_sha256_count is exact when verify_split_content=true.",
                },
            }
        return manifest

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name== "cifar224":
        return iCIFAR224()
    elif name== "imagenetr":
        return iImageNetR()
    elif name=="imageneta":
        return iImageNetA()
    elif name=="cub":
        return CUB()
    elif name=="vtab":
        return vtab()
    elif name=="car196":
        return iCARS196_224()
    elif name=="omnibenchmark":
        return omnibenchmark()
    elif name=="objectnet":
        return objectnet()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
