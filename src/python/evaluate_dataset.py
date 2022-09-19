from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from scipy import stats
import nibabel as nib

# calculate p-value from auc using permutation
def calc_pval_auc_permutation(y, x, auc, nsamples=1000):
    auc_permut = np.empty(nsamples)
    idx_p = np.arange(len(y))
    for p in range(nsamples):
        np.random.shuffle(idx_p)
        roc_auc = roc_auc_score(y[idx_p], x)
        auc_permut[p] = roc_auc
    pval_permut = np.mean(auc_permut>=auc)
    return np.mean(auc_permut), pval_permut

def evaluate_rois_likelihoods(args, df, fields_eval, scores=True):
    dataset_name = args.dataset_name
    dataset_prefix = 'args.dataset_info.' + dataset_name
    common_prefix = 'args.common_files'
    likelihood_paths = eval(dataset_prefix + '.likelihood_paths')
    pvalue_corrected_alpha = eval(eval(dataset_prefix + '.pvalue_corrected_alpha'))
    nii_map_file = eval(common_prefix + '.nii_map_file')
    nii_atlas_file = eval(common_prefix + '.nii_atlas_file')
    nii_icv_mask = eval(common_prefix + '.nii_icv_mask')
    likelihood_roi_paths = [f"{x}selected_probs/" for x in likelihood_paths]
    info_atlas_file = eval(common_prefix + '.info_atlas_file')
    df_atlas = pd.read_csv(info_atlas_file, usecols=[0, 1], sep=' ', header=None, names=['id', 'name'])

    # load map and atlas
    map_idx = nib.load(nii_map_file)
    map_idx_np = map_idx.get_fdata().astype(int)
    # map_idx_affine = map_idx.affine
    atlas_idx = nib.load(nii_atlas_file)
    atlas_idx_np = atlas_idx.get_fdata().astype(int)
    # atlas_idx_affine = atlas_idx.affine
    mask_icv = nib.load(nii_icv_mask)
    mask_icv_np = mask_icv.get_fdata().astype(int)
    # mask_icv_affine = mask_icv.affine

    # store the mean likelihoods of all transformers in all_subjects_probs_mean
    print("")
    for model_num, path in enumerate(likelihood_roi_paths):
        print(f"       -- Reading data to produce ensemble - model {model_num}")
        for i in range(len(df)):
            # if i % 500 == 0:
                # print(f"Processing row {i}...")
            selected_subject_npy = f"{path}ord_selected_probs_{i}.npy"
            subject_probs = np.load(selected_subject_npy, allow_pickle=True)
            if i == 0:
                all_subjects_probs = subject_probs
            else:
                all_subjects_probs = np.concatenate((all_subjects_probs, subject_probs), axis=0)
        if model_num == 0:
            all_subjects_probs_sum = all_subjects_probs
        else:
            all_subjects_probs_sum += all_subjects_probs
    all_subjects_probs_mean = all_subjects_probs_sum / (model_num + 1)

    ### calculate mean intracranial volume likelihoods
    # create log_all_subjects_probs_mean variable
    log_all_subjects_probs_mean = np.zeros(shape=all_subjects_probs_mean.shape)
    for i in range(len(all_subjects_probs_mean)):
        log_all_subjects_probs_mean[i, :] = np.log(all_subjects_probs_mean[i, :])
        # get 3d indices for all intra cranial volume mask
    icv_indexes_3d = (mask_icv_np * map_idx_np).nonzero()
    # get corresponding indexes for serialized tokens
    icv_indexes_tokens = map_idx_np[icv_indexes_3d]
    # get mean probability of all intra cranial volume tokens
    icv_mean_log_probs = log_all_subjects_probs_mean[:, icv_indexes_tokens].mean(axis=1)

    # test each region (atlas_id) of brain for correlation
    print(f"\n       p-value corrected alpha: {eval(dataset_prefix + '.pvalue_corrected_alpha')}")
    if scores:
        print("\n       field,atlas_id,atlas_descr,r-pearson,p-value")
    else:
        print("\n       field,atlas_id,atlas_descr,auc_roc,p-value")

    for atlas_id in set(atlas_idx_np.flatten()):
        # does not evaluate background due to memory overflow
        if atlas_id != 0:
            # bool mask for atlas id
            mask_id = (atlas_idx_np == atlas_id)
            # get corresponding 3d indices from map
            indexes_3d = (mask_id * map_idx_np).nonzero()
            # get corresponding index for serialized tokens
            indexes_tokens = map_idx_np[indexes_3d]
            # get mean probability from the atlasid indexes of tokens
            mean_probs_atlasid = all_subjects_probs_mean[:, indexes_tokens].mean(axis=1)
            # test correlation for each evaluation field
            for field in fields_eval:
                y = df[field].values
                x = mean_probs_atlasid
                atlas_descr = df_atlas[df_atlas['id'] == atlas_id]['name'].values[0]
                if scores:
                    r, pval = stats.pearsonr(x, y)
                    if pval < pvalue_corrected_alpha:  # p-value corretion for <0.05
                        print(f"       {field},{atlas_id},{atlas_descr},{r},{pval:.2e}")
                else:
                    auc = roc_auc_score(y, 1 - x)  # use 1-x, as lower probabilities represent higher chances of autism
                    auc_permut, pval_permut = calc_pval_auc_permutation(y, 1 - x, auc, nsamples=1000)
                    if pval_permut < pvalue_corrected_alpha:  # p-value corretion for <0.05
                        print(f"       {field},{atlas_id},{atlas_descr},{auc},{pval_permut:.2e}")

def main(args):
    # print(args)
    dataset_name = args.dataset_name
    dataset_prefix = 'args.dataset_info.' + dataset_name

    participants_tsv = eval(dataset_prefix + '.participants_tsv')
    likelihood_paths = eval(dataset_prefix + '.likelihood_paths')
    score_fields_eval = eval(dataset_prefix + '.score_fields_eval')
    diagnostic_fields_eval = eval(dataset_prefix + '.diagnostic_fields_eval')
    random_seed = eval('args.random_seed')

    np.random.seed(random_seed)

    df_test = pd.read_csv(participants_tsv, sep='\t', header=0)
    df_test = df_test.replace(-999, np.NaN)

    for i, path in enumerate(likelihood_paths):
        likelihood_npy = path + 'likelihood.npy'
        likelihood_tensors = np.load(likelihood_npy, allow_pickle=True)
        likelihood_values = list(map(lambda x: x.numpy()[0], likelihood_tensors))
        df_test[f"likelihood_{i}"] = likelihood_values

    print('\nEvaluating dataset: ' + dataset_name)

    emsemble_list = []
    for i in range(9):
        x = df_test[f'likelihood_{i}']
        emsemble_list.append(x)
    print(f'\n-- Total Likelihood Metrics (Transformers emsemble):')
    emsemble_x = np.array(emsemble_list).mean(axis=0)

    if score_fields_eval is not None:
        df_test_scores = df_test[score_fields_eval].copy()
        df_test_scores['emsemble_x'] = emsemble_x
        df_test_scores = df_test_scores.dropna().reset_index()
        x = df_test_scores['emsemble_x']
        n = len(x)
        print(f'\n   -- Evaluating score keys. Sample size N={n}.')
        for field in score_fields_eval:
            y = df_test_scores[field]
            r, pval = stats.spearmanr(x, y)
            print(f'{field: >30} | spearman-r: {r:+0.2f}, pval: {pval:0.4e}')
        evaluate_rois_likelihoods(args, df_test_scores, score_fields_eval, scores=True)
    if diagnostic_fields_eval is not None:
        df_test_diagnostics = df_test[diagnostic_fields_eval].copy()
        df_test_diagnostics['emsemble_x'] = emsemble_x
        df_test_diagnostics = df_test_diagnostics.dropna().reset_index()
        x = df_test_diagnostics['emsemble_x']
        n = len(x)
        print(f'\n   -- Evaluating diagnostic keys. Sample size N={n}.')
        for field in diagnostic_fields_eval:
            y = df_test_diagnostics[field]
            auc = roc_auc_score(y, x.abs())
            auc_permut, pval_permut = calc_pval_auc_permutation(y, x.abs(), auc, nsamples=1000)
            print(f'{field: >30} |        auc: {auc:+0.2f}, pval: {pval_permut:0.4f}')
        evaluate_rois_likelihoods(args, df_test_diagnostics, diagnostic_fields_eval, scores=False)

if __name__ == "__main__":
    args_cfg = OmegaConf.load("/project/src/python/config/evaluate_dataset.yaml")
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_cfg, args_cli)

    main(args)