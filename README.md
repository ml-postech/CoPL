# CoPL: Collaborative Preference Learning

This repository contains the official implementation of the paper:

**CoPL: Collaborative Preference Learning for Personalizing LLMs**  
Youngbin Choi, Seunghyuk Cho, Minjong Lee, MoonJeong Park, Yesong Ko, Jungseul Ok, Dongwoo Kim  
*arXiv preprint, 2025*

**Paper**: [arXiv:2503.01658](https://arxiv.org/abs/2503.01658)

CoPL is a collaborative preference learning framework that learns user preferences and adapts to new users. This project consists of 4 stages:

1. **Dataset Generation** - Generate user preference data
2. **User Representation Learning** - Learn user embeddings
3. **RM Training** - Train reward models
4. **Unseen Adaptation** - Adapt to new users

## Requirements

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install custom PEFT package with MoLE support
cd peft-main
pip install -e .
cd ..
```

## Usage

### 1. Dataset Generation

Generate user preference data using the following commands:

```bash
# PLM dataset
python data/datagen_plm.py --tokenizer_name google/gemma-2b-it --num_users 10000 --n_context 16 --seed 1111 
python data/datagen_plm.py --tokenizer_name google/gemma-2b-it --num_users 10000 --n_context 16 --seed 1111 --AVG

# TLDR dataset
python data/datagen_tldr.py --tokenizer_name google/gemma-2b-it --num_users 10000 --n_context 8 --seed 1111 
python data/datagen_tldr.py --tokenizer_name google/gemma-2b-it --num_users 10000 --n_context 8 --seed 1111 --AVG

# UltraFeedback-P dataset
python data/datagen_ufp.py --other_subsets UF-P-2 --tokenizer_name google/gemma-2b --model_name google/gemma-2b --num_users 10000
python data/datagen_ufp.py --other_subsets UF-P-4 --tokenizer_name google/gemma-2b --model_name google/gemma-2b --num_users 10000
python data/datagen_ufp.py --other_subsets UF-P-2 --tokenizer_name google/gemma-2b --model_name google/gemma-2b --num_users 10000 --AVG 
python data/datagen_ufp.py --other_subsets UF-P-4 --tokenizer_name google/gemma-2b --model_name google/gemma-2b --num_users 10000 --AVG 
```

The data is stored in the following format:

```python
# Data format
{
    'user': user_id,
    'context': [(positive_item, negative_item), ...],
    'context_unseen': [(positive_item, negative_item), ...],
    'target': [(positive_item, negative_item), ...],
    'user_type': user_type
}
```

### 2. User Representation Learning

Learn user embeddings using the CoPLGCF model:

```bash
python train_CoPL_gcf.py \
    --data_path dataset/your_data.pkl \
    --hidden_dim 512 \
    --l 4 \
    --num_epoch 100 \
    --learning_rate 1e-4 \
    --wandbon True
```


### 3. RM Training

Train personalized reward models using user embeddings:

```bash
bash scripts/CoPL_rm_train.sh
```

Or run directly:

```bash
torchrun --nproc_per_node=4 --master_port 4788 train_CoPL_rm.py \
    --model_name google/gemma-2b-it \
    --data_path dataset/UF-P-2-10000-ALL.pkl \
    --user_embeds_path gcf_user_embeds/UF-P-2-ALL-user_emb.pt \
    --log_dir logs/CoPL_RM \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --lora_r 8 \
    --num_experts 8 \
    --max_steps 1500 \
    --deepspeed scripts/ds_config.json
```


### 4. Unseen Adaptation

Perform adaptation for new users:

```bash
python unseen_user_adaptation.py \
    --data_path dataset/your_data.pkl \
    --unseen_data_path dataset/unseen_data.pkl \
    --model_path gcf_models/your_model.pt \
    --save_path unseen_embeddings.pt \
    --hidden_dim 512 \
    --l 4
```

**Adaptation Method:**
- User embedding generation using 2-hop neighbor information
- Inference based on preference patterns of existing users
- Embedding aggregation through softmax weighted averaging

## Project Structure

```
CoPL/
├── models/
│   ├── CoPL_gcf.py          # User representation learning model
│   └── CoPL_rm.py           # Reward model
|   └── baselines/           # TODO: add baseline models
├── scripts/
│   ├── CoPL_rm_train.sh     # RM training script
│   └── ds_config.json       # DeepSpeed configuration
├── train_CoPL_gcf.py        # User representation learning
├── train_CoPL_rm.py         # Reward model training
├── unseen_user_adaptation.py # New user adaptation
```



## Acknowledgement

Our implementation is inspired by and builds upon the following works:

- **[MoCLE](https://github.com/gyhdog99/MoCLE)**: Mixture of Cluster-conditional LoRA Experts for Vision-language Instruction Tuning. Our MoLE architecture implementation is based on their code.
- **[VPL-LLM](https://github.com/WEIRDLabUW/vpl_llm)**: Understanding Hidden Context in Preference Learning. Our preference learning framework is based on their code.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{choi2025copl,
  title={CoPL: Collaborative Preference Learning for Personalizing LLMs},
  author={Choi, Youngbin and Cho, Seunghyuk and Lee, Minjong and Park, MoonJeong and Ko, Yesong and Ok, Jungseul and Kim, Dongwoo},
  journal={arXiv preprint arXiv:2503.01658},
  year={2025}
}
```

## Contact

If you have questions or suggestions about the project, please create an issue.


