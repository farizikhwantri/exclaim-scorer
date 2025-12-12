from peft import LoraConfig, get_peft_model, TaskType

def load_lora_model(model, target_modules: list, task_type=TaskType.SEQ_CLS, device="cpu"):
    # #If only targeting attention blocks of the model
    # target_modules = ["q_proj", "v_proj"]

    # #If targeting all linear layers
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    default_target_modules = ["attention.self.query", "attention.self.key", 
                              "attention.self.value", "attention.output.dense", 
                              "intermediate.dense"]
    if target_modules is None:
        target_modules = default_target_modules

    # for name, _ in model.named_modules():
    #     print(name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()

    return model