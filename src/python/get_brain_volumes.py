import argparse
from pathlib import Path

import pandas as pd
import torch
from monai import transforms
from monai.data import CacheDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", help="Directory name to generate output results.")
    parser.add_argument("--subject_ids", help="Location of file with subject ids.")

    args = parser.parse_args()
    return args

def get_data_dicts(subject_ids):
    df = pd.read_csv(subject_ids, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
                {
                    "t1w": (
                        f"/data/{row['dataset']}/derivatives/ANTs_N4_RegistrationSynQuick/{row['participant_id']}/"
                        f"ses-01/anat/{row['participant_id']}_ses-01_space-MNI152NLin2009aSym_T1w.nii.gz"),
                    "seg": (
                        f"/data/{row['dataset']}/derivatives/ANTs_N4_RegistrationSynQuick/{row['participant_id']}/"
                        f"ses-01/anat/{row['participant_id']}_ses-01_space-MNI152NLin2009aSym_wmparc.nii.gz")
                }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_data_loader(subject_ids: str):
    img_transforms = transforms.Compose(
        [
            transforms.LoadImaged(["t1w", "seg"]),
            transforms.AddChanneld(["t1w", "seg"]),
            transforms.ConcatItemsd(["t1w", "seg"], "image_seg"),
        ]
    )
    data_dicts = get_data_dicts(subject_ids)
    data_set = CacheDataset(
        data=data_dicts,
        transform=img_transforms
    )
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        num_workers=16,
        drop_last=False,
        pin_memory=True
    )
    return data_loader


def main(args):
    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)
    run_dir = output_dir / args.run_dir
    if not run_dir.exists():
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    data_loader = get_data_loader(subject_ids=args.subject_ids)

    for i, X in enumerate(data_loader):
        print('Summing voxels in ROIs of image number:', i)
        full_image = X['image_seg'][0, 0, :]
        segment_mask = X['image_seg'][0, 1, :]
        brain_count = torch.count_nonzero(full_image * segment_mask).numpy()
        rois_dict = {'id_' + str(x): 0.0 for x in segment_mask.unique().int().numpy()}
        for roi_id in segment_mask.unique():
            roi_mask = (segment_mask == roi_id)
            roi_image = full_image * roi_mask
            roi_count = torch.count_nonzero(roi_image).numpy()
            roi_volume = roi_count / brain_count
            id_str = 'id_' + str(roi_id.int().numpy())
            rois_dict[id_str] = roi_volume
        if i == 0:  # first row
            df_feat = pd.DataFrame([rois_dict])
        else:  # other rows
            fields_add = set(rois_dict.keys()) - set(list(df_feat.columns))
            if len(fields_add) > 0:
                for id_str in fields_add:
                    df_feat[id_str] = 0.0
            df_feat = df_feat.append([rois_dict], ignore_index=True)

    print('Concatenating ids and features dataframes...')
    df_subjects = pd.read_csv(args.subject_ids, sep="\t")
    df = pd.concat([df_subjects, df_feat], axis=1)
    output_tsv = str(run_dir)+'/subjects_rois_features.tsv'
    print('Saving dataframe in tsv file: ' + output_tsv)
    df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
