# Normative modeling for psychiatry
Normative modeling for psychiatry based on deep learning models.

### 1. VBM preprocessing:

#### 1.1. Generate matlab spm12 scripts
 - Create a file named all_nii_files.txt containing all files to be VBM processed (file paths must be relative to the spm12 docker container).
 - Copy *.sh and *.m files from src/preprocessing/rawtovbm/ to the same directory of all_nii_files.txt.
 - Run the script generate_spm12_scripts.sh.
 - The generated scripts should look like the ones in:
   - src/preprocessing/rawtovbm/abcd/
   - src/preprocessing/rawtovbm/abcd_y2/
   - src/preprocessing/rawtovbm/bhrcs/
   - src/preprocessing/rawtovbm/adhd200/
   - src/preprocessing/rawtovbm/abide2/.  

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

#### 1.4. ABCD_Y2 preprocessing jobs:
```bash
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 0
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 1
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 2
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 3
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 4
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 5
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 6
runai/preprocessing/submit_ABCDY2_spm12_part1.sh 7

runai/preprocessing/submit_ABCDY2_spm12_part2.sh
```

#### 1.5. ADHD200 preprocessing jobs:
```bash
runai/preprocessing/submit_ADHD200_spm12_part1.sh 0

runai/preprocessing/submit_ADHD200_spm12_part2.sh
```

#### 1.6. ABIDE2 preprocessing jobs:
```bash
runai/preprocessing/submit_ABIDE2_spm12_part1.sh 0

runai/preprocessing/submit_ABIDE2_spm12_part2.sh
```

### 2. Training models:

#### 2.1. VQ-VAE
```bash
runai/training/submit_train_vqvae.sh
```

#### 2.2. Transformers
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

#### 3.1. ABCD
```bash

```

#### 3.2. BHRCS
```bash

```

#### 3.3. ADHD200
```bash

```

#### 3.4. ABIDE-II
```bash

```
