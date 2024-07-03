# Source codes for the manuscript: GPT-based normative models of brain sMRI correlate with dimensional psychopathology

### Please cite:
Sergio Leonardo Mendes, Walter Hugo Lopez Pinaya, Pedro Mario Pan, Ary Gadelha, Sintia Belangero, Andrea Parolin Jackowski, Luis Augusto Rohde, Euripedes Constantino Miguel, João Ricardo Sato; GPT-based normative models of brain sMRI correlate with dimensional psychopathology. Imaging Neuroscience 2024; 2 1–15. doi: https://doi.org/10.1162/imag_a_00204

Trackeable sources codes registered with Zenodo:
[10.5281/zenodo.10694519](https://doi.org/10.5281/zenodo.10694519)


### 1. VBM preprocessing:

#### 1.1. Generate matlab spm12 scripts
 - Create a file named all_nii_files.txt containing all files to be VBM processed (file paths must be relative to the spm12 docker container).
 - Copy *.sh and *.m files from src/preprocessing/rawtovbm/ to the same directory of all_nii_files.txt.
 - Run the script generate_spm12_scripts.sh.
 - The generated scripts should look like the ones in:
   ```bash
   src/preprocessing/rawtovbm/abcd/
   src/preprocessing/rawtovbm/bhrcs/
   src/preprocessing/rawtovbm/adhd200/
   src/preprocessing/rawtovbm/abide2/
   ```

#### 1.2. BHRCS preprocessing jobs:
```bash
runai/preprocessing/submit_BHRCS_spm12_part1.sh 0
runai/preprocessing/submit_BHRCS_spm12_part1.sh 1
runai/preprocessing/submit_BHRCS_spm12_part2.sh
```

#### 1.3. ABCD preprocessing jobs:
```bash
runai/preprocessing/submit_ABCD_spm12_part1.sh 0
runai/preprocessing/submit_ABCD_spm12_part1.sh 1
runai/preprocessing/submit_ABCD_spm12_part1.sh 2
runai/preprocessing/submit_ABCD_spm12_part1.sh 3
runai/preprocessing/submit_ABCD_spm12_part1.sh 4
runai/preprocessing/submit_ABCD_spm12_part1.sh 5
runai/preprocessing/submit_ABCD_spm12_part1.sh 6
runai/preprocessing/submit_ABCD_spm12_part1.sh 7
runai/preprocessing/submit_ABCD_spm12_part1.sh 8
runai/preprocessing/submit_ABCD_spm12_part1.sh 9
runai/preprocessing/submit_ABCD_spm12_part2.sh
```

#### 1.4. ADHD200 preprocessing jobs:
```bash
runai/preprocessing/submit_ADHD200_spm12_part1.sh 0
runai/preprocessing/submit_ADHD200_spm12_part2.sh
```

#### 1.5. ABIDE2 preprocessing jobs:
```bash
runai/preprocessing/submit_ABIDE2_spm12_part1.sh 0
runai/preprocessing/submit_ABIDE2_spm12_part2.sh
```

### 2. Training models:

#### 2.1. VQ-VAE job:
```bash
runai/training/submit_train_vqvae.sh
```

#### 2.2. Transformers jobs:
```bash
runai/training/submit_train_transformer_0.sh
runai/training/submit_train_transformer_1.sh
runai/training/submit_train_transformer_2.sh
runai/training/submit_train_transformer_3.sh
runai/training/submit_train_transformer_4.sh
runai/training/submit_train_transformer_5.sh
runai/training/submit_train_transformer_6.sh
runai/training/submit_train_transformer_7.sh
runai/training/submit_train_transformer_8.sh
```

### 3. Getting likelihood predictions:

#### 3.1. ABCD jobs:
```bash
runai/evaluation/submit_get_likelihood_0.sh
runai/evaluation/submit_get_likelihood_1.sh
runai/evaluation/submit_get_likelihood_2.sh
runai/evaluation/submit_get_likelihood_3.sh
runai/evaluation/submit_get_likelihood_4.sh
runai/evaluation/submit_get_likelihood_5.sh
runai/evaluation/submit_get_likelihood_6.sh
runai/evaluation/submit_get_likelihood_7.sh
runai/evaluation/submit_get_likelihood_8.sh
```

#### 3.2. BHRCS jobs:
```bash
runai/evaluation/bhrcs_submit_get_likelihood_0.sh
runai/evaluation/bhrcs_submit_get_likelihood_1.sh
runai/evaluation/bhrcs_submit_get_likelihood_2.sh
runai/evaluation/bhrcs_submit_get_likelihood_3.sh
runai/evaluation/bhrcs_submit_get_likelihood_4.sh
runai/evaluation/bhrcs_submit_get_likelihood_5.sh
runai/evaluation/bhrcs_submit_get_likelihood_6.sh
runai/evaluation/bhrcs_submit_get_likelihood_7.sh
runai/evaluation/bhrcs_submit_get_likelihood_8.sh
```

#### 3.3. ADHD200 jobs:
```bash
runai/evaluation/adhd200_submit_get_likelihood_0.sh
runai/evaluation/adhd200_submit_get_likelihood_1.sh
runai/evaluation/adhd200_submit_get_likelihood_2.sh
runai/evaluation/adhd200_submit_get_likelihood_3.sh
runai/evaluation/adhd200_submit_get_likelihood_4.sh
runai/evaluation/adhd200_submit_get_likelihood_5.sh
runai/evaluation/adhd200_submit_get_likelihood_6.sh
runai/evaluation/adhd200_submit_get_likelihood_7.sh
runai/evaluation/adhd200_submit_get_likelihood_8.sh
```

#### 3.4. ABIDE-II jobs:
```bash
runai/evaluation/abide2_submit_get_likelihood_0.sh
runai/evaluation/abide2_submit_get_likelihood_1.sh
runai/evaluation/abide2_submit_get_likelihood_2.sh
runai/evaluation/abide2_submit_get_likelihood_3.sh
runai/evaluation/abide2_submit_get_likelihood_4.sh
runai/evaluation/abide2_submit_get_likelihood_5.sh
runai/evaluation/abide2_submit_get_likelihood_6.sh
runai/evaluation/abide2_submit_get_likelihood_7.sh
runai/evaluation/abide2_submit_get_likelihood_8.sh
```

### 4. Evaluate datasets:

#### 4.1. Evaluate ABCD:
```bash
python src/python/evaluate_dataset.py dataset_name=ABCD
```

#### 4.2. Evaluate BHRCS
```bash
python src/python/evaluate_dataset.py dataset_name=BHRCS
```

#### 4.3. Evaluate ABIDE-II
```bash
python src/python/evaluate_dataset.py dataset_name=ABIDE2
```

#### 4.4. Evaluate ADHD-200
```bash
python src/python/evaluate_dataset.py dataset_name=ADHD200
```
