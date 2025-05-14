import wandb
wandb.login()


config = dict(project="mask_detection", 
              name="mask_detection_v1",
              epochs=10,
              batch_size=32,
              learning_rate=0.001,
              )

wandb.init(config=config)

#update config
with wandb.init(config=config) as run:
    run.config.update(config)









