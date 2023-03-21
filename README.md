# Step 1: Installation

VariPred was trained using the softwares included in the file "requirements.txt". Please install requirements first.

```shell
$ git@github.com:wlin16/VariPred.git
$ cd VariPred
$ conda create -n varipred python=3.8.5
$ conda install --file requirements.txt

```
=========================================================================================================

## Step 2: Prepare dataset

The input for VariPred must contains

1.  **mutation info** (e.g. **NP_001035957.1_L847P**), a.k.a "target_id"

2.  **mutation position**, a.k.a "aa_index"

3.  **wild-type amino acid**, a.k.a "wt_aa"

4.  **mutant-type amino acid**, a.k.a "mt_aa"

5.  **wild-type sequence**, a.k.a "wt_seq"

6.  **mutant-type sequence**, a.k.a "mt_seq"

    Label is optional for a prediction purpose but neccessary for preparing the training and test sets if re-train the model.

    Please name the column of the dataframe according to the example files under the "example" folder.

- For data with UniProt IDs, please fetch the wildtype with https://www.uniprot.org/id-mapping

- For data with RefSeq IDs (NP ids), please use the "prepare_dataset.py" script under the Dataset_preparation folder:

  - Inside Dataset_preparation folder, we prepared an example "target.txt". The first parameter is the name of the file.

    ```shell
    $ cd Dataset_preparation
    $ python3 prepare_dataset.py target
    ```

- Now, we have a dataframe named as "target.csv"

=========================================================================================================



## Step 3: Train the model

*Note: The weight of the trained model mentioned in the publication has been given under the directory of VariPred/model/model.ckpt. Running train_VariPred.sh script to re-train the model will replace the given weight. If there is no need to customize the model for a specific task, but only to use a trained model to make clinical impact predictions for variants, please skip this step and proceed directly to step 5.*  

*If there is no need to customize the model for a specific task or evaluate the performance of VariPred, but only to use VariPred to make clinical impact predictions for variants, please skip Step 3 and Step 4 and proceed directly to Step 5.*  



We recommand you have an at least 12GB GPU, e.g. NVIDIA GeForce 1080Ti

PyTorch should be installed, see: https://pytorch.org/get-started/locally/

"train.csv", "test.csv" are the example files as the training and test sets to re-train the model. "target.csv" is the example file for a simple prediction purpose.

- - To prepare a training set and a test set, you can prepare the datasets with the following codes

    ```shell
    $ python3 prepare_dataset.py VariPred_train
    $ python3 prepare_dataset.py VariPred_test
    ```

- Now, we have two dataframe named as "VariPred_train.csv" and "VariPred_test.csv" under the directory of example/dataset

=========================================================================================================


## Setp 4: Fetch the embeddings and train the model

1. If you would like to re-train the VariPred, the embedding representations need to be generated for both the training and test sets.

   ```shell
   $ cd ../VariPred
   ```

2. Replace the variables based on the paths where you stored your datasets in "train_VariPred.sh" script. Then run the script. This will give you the performance of the model (MCC and AUC-ROC scores). 

```shell
$ ./train_VariPred.sh
```

=========================================================================================================

## Setp 5: Fetch the embeddings and predict the effects of variants

1. To predict the effects of variants by VariPred, please replace the variables based on the paths where you stored your data in "predict.sh" script. Then run the script. This will give you the clinical impact of each variants

   ```shell
   $ cd ../VariPred
   $ ./predict.sh
   ```
