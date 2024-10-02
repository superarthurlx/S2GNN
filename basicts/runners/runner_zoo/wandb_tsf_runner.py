import os
import torch
import wandb

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import pdb

class WandBTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_name = cfg['MODEL']['NAME']
        self.dataset_name = cfg['DATASET_NAME']
        # os.environ["WANDB_API_KEY"]="8ca6de7fc78931238f3498855f894713df9c5e1d"
        wandb.init(
            # set the wandb project where this run will be logged
            project = "time series adp adj",
            name = self.model_name,
            # track hyperparameters and run metadata
            config = cfg
            # config={
            # "learning_rate": 0.02,
            # "architecture": "CNN",
            # "dataset": "CIFAR-100",
            # "epochs": 10,
            # }
        )

        wandb.watch(self.model, log="all")
        self.wdb = wandb

    def normalized_laplacian(self, adj):
        degree = torch.sum(adj, dim=1)
        adj = 0.5 * (adj + adj.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_l - adj, diagonal_degree_hat))
        return laplacian

    # # support test process
    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """
        # log metrics to wandb
        
        # self.wdb.log({"train_MAE": self.meter_pool.get_avg('train_MAE')})
        # print train meters
        self.print_epoch_meters("train")
        # tensorboard plt meters
        self.plt_epoch_meters("train", epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_process(train_epoch=epoch)
        # save model
        self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

        # =========================================

        # self.wdb.log({"epoch": epoch})

        # for i in range(10):
        #     self.wdb.log({"epoch": epoch, f"kernel_{i}":self.model.node_feature_connection.conv1.weight[i,0,0]})
        # pdb.set_trace()
        norm_adj = self.normalized_laplacian(self.model.adp)
        self.wdb.log({"%s"%self.dataset_name:norm_adj.norm(p='fro').item()})

    #     # # start a new wandb run to track this script
       

    #     # # simulate training
    #     # epochs = 10
    #     # offset = random.random() / 5
    #     # for epoch in range(2, epochs):
    #     #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     #     loss = 2 ** -epoch + random.random() / epoch + offset

    #     #     # log metrics to wandb
    #     #     wandb.log({"acc": acc, "loss": loss})

    #     # # [optional] finish the wandb run, necessary in notebooks
    #     # wandb.finish()