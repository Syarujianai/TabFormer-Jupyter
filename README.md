# Introduction

This project integrates the code from the [TabFormer](https://github.com/IBM/TabFormer) repo, in order to run it directly in JUPYTER NOTEBOOK

## cited 

```markdown
@inproceedings{padhi2021tabular,
	title="Tabular Transformers for Modeling Multivariate Time Series",
	author="Inkit {Padhi} and Yair {Schiff} and Igor {Melnyk} and Mattia {Rigotti} and Youssef {Mroueh} and Pierre {Dognin} and Jerret {Ross} and Ravi {Nair} and Erik {Altman}",
	booktitle="ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
	pages="3565--3569",
	notes="Sourced from Microsoft Academic - https://academic.microsoft.com/paper/3160590016",
	year="2021"
}
```

## features

- `TabFormer_predict.py`: get mean pooled embeddings of train set and test set,  and output as `.csv` file

## how to run

- Customize YOUR  `encode_data`   function related to `[YOUR DATASET]`

- Required  arguments:

  ```shell
  --data_fname transaction [YOUR DATASET name]
  --dataroot ./data [YOUR DATASET root]
  --save_steps 6500 [checkpoint save steps]
  --checkpoint [YOUR CHEKPOINT]
  --max_truncate_row 100 [maximum number of transactions for single user]
  --n_layers 2  [number of transformer blocks]
  --mlm [mask language model loss]
  --field_ce [cross-fielded transaction embedding]
  --user_ids [specified users for Evaluation]
  --do_train [Training or not]
  ```

- For model pretraining:

  ```shellpython ./TabFormer.py
  python ./TabFormer_pretrain.py
  ```

- For loading raw TabFormer embeddings:

  ```python
  python ./TabFormer_predict.py
  ```

  - embeddings are written into `/outputs/checkpoint-{args.checkpoint}-eval.csv`
  - user id indices of written `.csv`  embeddings are duplicate, which needs to apply follow steps:

  ```python
  embeddings = pd.read_csv("checkpoint-{args.checkpoint}-eval.csv")
  embeddings = embeddings.drop_duplacates("user_id")
  ```
  - NOTE: zero filling is applied for the user has no transactions 

# TODOs

- CPU acceleration for `transformers`