from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import random
import logging
import pickle

import numpy as np
import torch
from torch.utils.data import Subset
import torch.distributed

import peft
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
    TrainerCallback,
)
from transformers.utils import PaddingStrategy
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from models.CoPL_rm import CoPLRMTrainer, CoPLRM

@dataclass
class ScriptArguments:
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed."
        },
    )
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=3e-6)
    weight_decay: float = field(default=0.001)
    lora_r: int = field(default=8)
    num_experts: int = field(default=4)
    num_user_per_batch: int = field(default=8)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "Model name from Hugging Face hub (e.g. gpt2, gpt2-xl, bert)"
        },
    )
    data_path: str = field(default="LGCN-l4embeds.pt")
    user_embeds_path: str = field(default="Anthropic/hh-rlhf")
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name, defaults to model name if not specified"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use float16 precision"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
    )
    max_length: int = field(default=1024)
    log_dir: str = field(default="data/reward_models/hh_rlhf")
    latent_dim: int = field(
        default=512,
        metadata={"help": "Dimension of latent user vector"}
    )
    hidden_dim: int = field(
        default=512,
        metadata={"help": "Dimension of hidden layer in VAE"}
    )
    encoder_embed_dim: int = field(
        default=1024,
        metadata={"help": "Dimension of LLM embeddings for encoder"}
    )
    decoder_embed_dim: int = field(
        default=1024,
        metadata={"help": "Dimension of LLM embeddings for decoder"}
    )
    wandbon: bool = field(default=False)
    seed: int = field(default=0)
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    max_steps: int = field(
        default=3000,
        metadata={"help": "Maximum number of training steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between evaluations"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between model saves"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between logging"}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )

def get_step_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    if current_step < num_training_steps // 3:
        return 1.0
    elif current_step < (2 * num_training_steps) // 3:
        return 0.1
    else:
        return 0.01


def get_cosine_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * current_step / num_training_steps))


@dataclass
class CoPLRMDataCollatorWithPadding:
    args: ScriptArguments
    tokenizer: PreTrainedTokenizerBase
    total: List[Dict[str, torch.Tensor]]
    E_u: torch.Tensor
    num_user: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features):

        u_embeds_list = []
        text_features = []
        n_text_features = []
        controversial_list = []
        user_type_list = []
            
        for feature in features:
            u_id, pos, neg, user_type = feature
            u_embeds_list.append(self.E_u[u_id])
            pos_data = self.total[pos]
            neg_data = self.total[neg]
            user_type_list.append(int(user_type))
            controversial_list.append(False) # NOTE: you need to pass the controversial label here if you want to use it

            text_features.append({
                "input_ids": pos_data['ids'],
                "attention_mask": pos_data['mask'],
            })
            n_text_features.append({
                "input_ids": neg_data['ids'],
                "attention_mask": neg_data['mask'],
            })
        
        batch = self.tokenizer.pad(
                text_features + n_text_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        
        return {        
            'input_ids': batch["input_ids"][:, :self.max_length],
            'input_mask': batch["attention_mask"][:, :self.max_length],
            'user_type': torch.tensor(user_type_list),
            'user_embeds': torch.stack(u_embeds_list+u_embeds_list),
            'controversial': torch.tensor(controversial_list),
            'return_loss': True
        }


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    # Set random seeds
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    torch.cuda.manual_seed(script_args.seed)

    torch.set_default_dtype(torch.bfloat16 if script_args.bf16 else torch.float32)

    # Set embedding dimensions based on model
    if script_args.model_name == 'gpt2':
        script_args.decoder_embed_dim = 768
        script_args.encoder_embed_dim = 768
    elif script_args.model_name == 'meta-llama/Llama-2-7b-hf':
        script_args.decoder_embed_dim = 4096
        script_args.encoder_embed_dim = 4096
    elif script_args.model_name == 'google/gemma-2b-it':
        script_args.decoder_embed_dim = 2048
        script_args.encoder_embed_dim = 2048
    elif script_args.model_name == 'google/gemma-7b-it':
        script_args.decoder_embed_dim = 3072
        script_args.encoder_embed_dim = 3072
    elif script_args.model_name == 'google/gemma-2-27b-it':
        script_args.decoder_embed_dim = 4608
        script_args.encoder_embed_dim = 4608

    # Load data
    total, users_context_target = pickle.load(open(script_args.data_path, 'rb'))
    
    # Calculate user weights
    uids_to_num_context = {i['user']: len(i['context']) for i in users_context_target}
    weights = np.array([uids_to_num_context[uid] for uid in list(uids_to_num_context.keys())], dtype=np.float64)
    weights /= weights.sum()

    # Load embeddings
    E_u = torch.load(script_args.user_embeds_path)
    E_u = E_u.detach().cpu()

    num_user = len(E_u)
    validtest_list=[]
    train_list=[]

    for user_context_target in users_context_target:
        user = user_context_target['user']
        target = user_context_target['target']
        
        train_list.extend([user, idx[0], idx[1], user_context_target['user_type']] for idx in user_context_target['context'] + user_context_target['context_unseen'])
        temp = np.random.choice(len(target), 11, replace=False).tolist()
        validtest_list.extend([user, target[idx][0], target[idx][1], user_context_target['user_type']] for idx in temp)
        
    # Split validation/test
    num_samples = len(validtest_list)
    subset_size = int(num_samples * 0.1)
    indices = random.sample(range(num_samples), subset_size)
    other_indices = list(set(range(num_samples)) - set(indices))

    valid_dataset = Subset(validtest_list, indices)
    test_dataset = Subset(validtest_list, other_indices)
    
    # Configure output path
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"{script_args.log_dir}/"
        f"{model_name_split}_r{script_args.lora_r}"
        f"_{script_args.user_embeds_path}_seed{script_args.seed}"
    )

    # Configure learning rate scheduler
    trainer_kwargs: Dict[str, Any] = {}
    if script_args.lr_scheduler_type == "step":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_step_decay_lr_lambda
    elif script_args.lr_scheduler_type == "cosine":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_cosine_decay_lr_lambda
    else:
        lr_scheduler_type = script_args.lr_scheduler_type

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        max_steps=script_args.max_steps,
        eval_steps=script_args.eval_steps,
        save_total_limit=script_args.save_total_limit,
        save_strategy="steps",
        metric_for_best_model='Accuracy',
        load_best_model_at_end=True,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        logging_strategy="steps",
        logging_steps=script_args.logging_steps,
        optim=script_args.optim,
        lr_scheduler_type=lr_scheduler_type,
        report_to='wandb' if script_args.wandbon else 'none',
        run_name=script_args.data_path + str(script_args.num_experts),
        dataloader_num_workers=script_args.num_workers,
    )
    
    # Load tokenizer
    tokenizer_name = script_args.tokenizer_name or script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=True, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Configure LoRA
    r = script_args.lora_r
    #########################################################
    # NOTE: MoLE configuration
    peft_config = LoraConfig(
                inference_mode=False, 
                r=r, 
                lora_alpha=2*r, 
                lora_dropout=0.1,
                )
    peft_config.multiple_loras=True
    peft_config.noise_std=0.1
    peft_config.gates_tmp=1.0
    peft_config.topk=1
    peft_config.num_experts=script_args.num_experts
    peft_config.cluster=True
    peft_config.g_enable=True
    #########################################################
    
    torch.set_anomaly_enabled(True)

    decoder_embed_dim = script_args.decoder_embed_dim
    encoder_embed_dim = script_args.encoder_embed_dim
    
    model = AutoModel.from_pretrained(
        script_args.model_name, torch_dtype=torch.bfloat16)
    
    model = get_peft_model(model, peft_config, adapter_name='0')
    
    # Add additional adapters
    for i in range(script_args.num_experts-1):
        model.add_adapter(str(i+1), peft_config)
    if peft_config.g_enable:
        model.add_adapter("g", peft_config)

    model.print_trainable_parameters()
    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    # Create reward model
    reward_model = CoPLRM(script_args.encoder_embed_dim, model)

    trainer = CoPLRMTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_list,
        eval_dataset=valid_dataset,
        compute_metrics=CoPLRMTrainer.compute_metrics,
        data_collator=CoPLRMDataCollatorWithPadding(
            args=script_args,
            total = total,
            E_u = E_u,
            num_user = script_args.num_user_per_batch,
            tokenizer=tokenizer,
            pad_to_multiple_of=64,
            max_length=script_args.max_length,),
        **trainer_kwargs,
    )

    # Add callback for first step evaluation
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(script_args.resume_from_checkpoint)
    test_result = trainer.evaluate(test_dataset, metric_key_prefix='test')
    print(test_result)
    