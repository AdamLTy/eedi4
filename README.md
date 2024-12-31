# Docker
- docker image for training: docker/Dockerfile, docker/build.sh
- docker image for data generation (vllm): docker_vllm/Dockerfile, docker_vllm/build.sh

# Data generation with Qwen2.5-72B-Instruct-AWQ
You can generate the data with the following command:
```
SEED=777
python src/gen/generate_train_72b_awq_100_example.py --seed ${SEED}
```

`results/train_gen/train_gen_8k.csv` is generated with 8 different seeds.

# Retriever training
## 32B baseline
### Training
```
for FOLD in 0 1; do
    python src/exp/exp010_fold_${FOLD}.py
done
```
### Inference for generated data
```
for FOLD in 0 1; do
    python src/exp/exp010_infer_gen_fold_${FOLD}.py
done
```
The output candidates `results/exp/ex010_infer_gen_fold_${FOLD}/exp010_infer_gen_fold_${FOLD}_val_pred.parquet` are used for the reranker training.
## 32B model training used in the final submission
```
for FOLD in 0 1; do
    python src/exp/exp012_fold_${FOLD}.py
done
```

# Reranker training
## 32B model training with the competition data and generated data
```
for FOLD in 0 1; do
    python src/exp/exp015_fold_${FOLD}.py
done
```
