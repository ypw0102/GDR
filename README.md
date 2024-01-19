#  Generative Dense Retrieval: Memory Can Be a Burden


## Environemnt
```bash
conda env create -f environment.yml
conda activate GDR
```


## Data Process

[1] Dataset Download.
Download NQ Train and Dev dataset from https://ai.google.com/research/NaturalQuestions/download
NQ Train: https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz
NQ Dev: https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz 
Please download it before re-training.

[2] Data preprocess 
You can process data with NQ_process.py (./Data_process/NQ_dataset/NQ_preprocess)


[3] Query Generation

In our study, Query Generation can significantly improve retrieve performance, especially for long-tail queries.

GDR uses [docTTTTTquery](https://github.com/castorini/docTTTTTquery) checkpoint to generate synthetic queries. If you finetune [docTTTTTquery](https://github.com/castorini/docTTTTTquery) checkpoint, the query generation files can make the retrieval result even better. We show how to finetune the model. The following command will finetune the model for 4k iterations to predict queries. We assume you put the tsv training file in gs://your_bucket/qcontent_train_512.csv (download from above). Also, change your_tpu_name, your_tpu_zone, your_project_id, and your_bucket accordingly.

```
t5_mesh_transformer  \
  --tpu="your_tpu_name" \
  --gcp_project="your_project_id" \
  --tpu_zone="your_tpu_zone" \
  --model_dir="gs://your_bucket/models/" \
  --gin_param="init_checkpoint = 'gs://your_bucket/model.ckpt-1004000'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/qcontent_train_512.csv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1008000" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v2-8'"
 ```

Please refer to docTTTTTquery documentation. 

Find more details in [NQ_dataset_Process.ipynb](./Data_process/NQ_dataset/NQ_dataset_Process.ipynb) and [Trivia_dataset_Process.ipynb](./Data_process/Trivia_dataset/Trivia_dataset_Process.ipynb).




## Training

Once the data pre-processing is complete, you can launch training by [train.sh](GDR_model/train.sh)

## Evaluation
Please use [infer.sh](GDR_model/infer.sh) along with checkpoint(Download it to './GDR_model/logs/'). You can also inference with your own checkpoint to evaluate model performance.

## Acknowledgement

We learned a lot and borrowed some code from the following projects when building GDR.

- [Transformers](https://github.com/huggingface/transformers)
- [docTTTTTquery](https://github.com/castorini/docTTTTTquery) 
- [NCI](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)
