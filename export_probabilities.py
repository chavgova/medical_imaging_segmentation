#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from torch.utils.data import DataLoader, Subset  
from dataset import SliceDataset 
from main import datasets_params, gt_transform, img_transform, make_net
from utils import tqdm_  


def patient_and_z(stem: str) -> tuple[str, int]:
    patient_id, z = stem.rsplit("_", 1)
    return patient_id, int(z)

 
def restore_model(result_dir: Path, training_config: dict, device: torch.device):
    dataset_name = training_config["dataset"] 
    model_name = training_config.get("model") or datasets_params[dataset_name]["model"]
    context_slices = int(training_config.get("context_slices", 0))
    
    model = make_net(model_name,
        2 * context_slices + 1,
        datasets_params[dataset_name]["K"],  
        training_config.get("foundation_model"),
        training_config.get("foundation_channels"),  
        training_config.get("foundation_fusion"), 
        int(training_config.get("foundation_upsample", 1)) )
    
    model.load_state_dict(torch.load(result_dir / "bestweights.pt", map_location="cpu"))
    
    return model.to(device).eval()
 

def resize_probabilities(probabilities: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = np.stack([
        resize(class_probability, shape, order=1, mode="constant", preserve_range=True, anti_aliasing=False)
        for class_probability in probabilities ], axis=-1).astype(np.float32)
    resized /= resized.sum(axis=-1, keepdims=True)
    
    return resized 

 
def export_patient(patient_id: str, indices: list[int], dataset: SliceDataset,
                   model, device: torch.device, class_count: int, batch_size: int,
                   image_pattern: str, probability_pattern: str) -> None:
    source_image = nib.load(image_pattern.format(id_=patient_id))  
    
    x, y, z = source_image.shape 
    probability_volume = np.zeros((x, y, z, class_count), dtype=np.float32) 
     
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)  

    with torch.no_grad():
        for data in loader:
            probabilities = F.softmax(model(data["images"].to(device)), dim=1).cpu().numpy()
            for slice_probabilities, stem in zip(probabilities, data["stems"]):
                _, slice_index = patient_and_z(stem)
                probability_volume[:, :, slice_index, :] = resize_probabilities(
                    slice_probabilities, (x, y)) 
 
    destination = Path(probability_pattern.format(id_=patient_id))
    destination.parent.mkdir(parents=True, exist_ok=True)  
    header = source_image.header.copy()  
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(probability_volume, source_image.affine, header), destination)


def main(args: argparse.Namespace) -> None:
    with open(args.result_dir / "config.json") as config_file:
        training_config = json.load(config_file)
    with open(args.postprocessing_config) as config_file:
        dense_crf_config = json.load(config_file)["dense_crf"]

    dataset_name = training_config["dataset"]
    saved_data_dir = training_config.get("data_dir")
    data_dir = args.data_dir or (Path(saved_data_dir) if saved_data_dir else Path("data") / dataset_name)
    class_count = datasets_params[dataset_name]["K"]
    context_slices = int(training_config.get("context_slices", 0))
    
    dataset = SliceDataset("val", data_dir,img_transform=img_transform, 
                           gt_transform=partial(gt_transform, class_count), context_slices=context_slices)

    indices_by_patient: dict[str, list[int]] = defaultdict(list)
    for index, (image_path, bla) in enumerate(dataset.files):
        patient_id, blabla = patient_and_z(image_path.stem)
        indices_by_patient[patient_id].append(index)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = restore_model(args.result_dir, training_config, device)
    batch_size = datasets_params[dataset_name]["B"]
    for patient_id, indices in tqdm_(indices_by_patient.items()):
        export_patient( patient_id, indices, dataset, model, device, class_count, batch_size, 
                       dense_crf_config["image_pattern"], dense_crf_config["probability_pattern"] )


def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="export validation softmax probabilities from completed model run")
    parser.add_argument("--result_dir", type=Path, required=True,
                        help="model result directory containing config.json and bestweights.pt")
    parser.add_argument("--postprocessing_config", type=Path,
                        default=Path(__file__).with_name("postprocessing_config.json"))
    parser.add_argument("--data_dir", type=Path,
                        help="for override of processed data_dir saved in the training config.")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
