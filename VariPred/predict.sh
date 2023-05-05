# Replace the following variables based on the paths where you store models and data
# df_path: folder name, where you store your data
# target_ds: dataframe name, the name of the data set you would like to get prediction (in .csv format)
# output_path: file name for the predicted results (in .txt format)

# In this example, we are using the training set named as "target.csv" under the directory "../example/dataset"
df_path="../example/dataset"
target_ds="erbb4varipred"
output_name="VariPred_output" 

python3 main.py \
                -p ${df_path} \
                -i ${target_ds} \
                -o ${output_name}