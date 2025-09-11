from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# initialise the model and tokenizer
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)


# Finetuning starts here

# Let's define the finetuning config and create the finetuning model
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Need to transform our jsonl dataset into the transform-expected format
data = load_dataset("json", data_files="./datasets/ft_toxic_dataset.jsonl", split="train")
data = data.map(lambda samples: tokenizer(samples["turn_resp"]), batched=True)
data = data.train_test_split(test_size=0.001)

# create finetuning trainer object
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    #    eval_dataset=data['test'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=400,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("alpharedteam")
tokenizer.save_pretrained("alpharedteam")