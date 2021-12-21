import wandb

sweep_config_t5 = {
    'name': 'T5 sweep',
    'method': 'grid',
    'parameters': {
        'model_type': {'value': 'T5'},
        'batch_size': {'value': 16},
        'epochs': {'value': 10},
        'num_runs': {'value': 1},
        'early_stop': {'value': 10},
        'learning_rate': {'values': [1e-4, 3e-4, 1e-3]},
        'data_dir': {'value': '/content/absa-bert'},
        'save_dir': {'value': '/content/drive/MyDrive'},
    }
}

sweep_id = wandb.sweep(sweep_config_t5, project="absa-enterpret", entity="saumb7800")
print(sweep_id)