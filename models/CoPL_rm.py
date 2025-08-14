import torch
import torch.nn as nn

from transformers import Trainer, EvalPrediction
from transformers.optimization import get_cosine_schedule_with_warmup

import peft


def set_lora_task_emb(peft_model, task_emb):
    for module in peft_model.model.modules():
        if isinstance(module, peft.tuners.lora.LoraLayer):
            module.input_emb = task_emb

class CoPLRM(nn.Module):
    def __init__(self, RM_embed_dim, llm):
        super(CoPLRM, self).__init__()
        
        self.llm = llm
        self.config = llm.config
        self.head = nn.Linear(RM_embed_dim, 1)

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)
        try:
            self.llm.base_model.wte.requires_grad_()
        except:
            self.llm.base_model.model.embed_tokens.requires_grad_()
    
    def forward(self, user_emb, input_ids, attention_mask):
        
        set_lora_task_emb(self.llm, user_emb)

        hidden_representations = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        token_length = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        representations = hidden_representations[:,token_length].diagonal().T
        reward = self.head(representations)

        return reward

class CoPLRMTrainer(Trainer):
    def __init__(
        self,
        lr_lambda=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda


    def compute_loss(self, wrapped_model, inputs, return_outputs=False, num_items_in_batch=None):
        if isinstance(wrapped_model, CoPLRM):
            model = wrapped_model
        else:
            model = wrapped_model.module

        user_embeds = inputs["user_embeds"]
        input_ids = inputs["input_ids"].cuda()
        input_mask = inputs["input_mask"].cuda()

        
        predicted_reward = model(user_emb = user_embeds, input_ids = input_ids, attention_mask = input_mask)
                
        controversial = inputs["controversial"]
        chosen_reward, rejected_reward = (predicted_reward).view(2, -1)

        loss = -(chosen_reward-rejected_reward).sigmoid().log().mean()

        if not return_outputs:
            self.log({"loss": loss.item()})
        else:
            return loss, {
                "user_type": inputs["user_type"],
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward}

        return loss
    

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler
    
    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        user_type, chosen_reward, rejected_reward = (eval_prediction.predictions)      

        chosen_reward = torch.from_numpy(chosen_reward)
        rejected_reward = torch.from_numpy(rejected_reward)
        user_type = torch.from_numpy(user_type)
        majority = user_type == 8

        acc = (chosen_reward > rejected_reward).float()
        con_acc = acc
        majority_acc = acc[majority]
        minority_acc = acc[~majority]

        return {"Accuracy": acc.mean(),
                "Controversial_acc": con_acc.mean(),
                "Majority_acc": majority_acc.mean(),
                "Minority_acc": minority_acc.mean()}
