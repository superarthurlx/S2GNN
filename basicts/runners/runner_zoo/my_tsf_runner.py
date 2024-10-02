import torch
import torchprofile
import wandb

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import pdb

class MyTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for multivariate time series forecasting datasets.
    Features:
        - Evaluate at pre-defined horizons (1~12 as default) and overall.
        - Metrics: MAE, RMSE, MAPE. Allow customization. The best model is the one with the smallest mae at validation.
        - Support setup_graph for the models acting like tensorflow.
        - Loss: MAE (masked_mae) as default. Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        
        self.c_flops = True 

        # 8ca6de7fc78931238f3498855f894713df9c5e1d
        self.model_name = cfg['MODEL']['NAME']
        self.dataset_name = cfg['DATASET_NAME']
        self.num_nodes = cfg['NUM_NODES']

        self.if_record_adj = cfg.get("RECORD_ADJ", False)

        if self.if_record_adj:
            self.a5 = torch.diag(torch.ones(self.num_nodes))
            self.a10 = torch.diag(torch.ones(self.num_nodes))
            self.a20 = torch.diag(torch.ones(self.num_nodes))
            self.a30 = torch.diag(torch.ones(self.num_nodes))
            self.a40 = torch.diag(torch.ones(self.num_nodes))

        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project = "Adaptive ADJ",
        #     name = self.model_name,
        #     # track hyperparameters and run metadata
        #     config = cfg
        #     # config={
        #     # "learning_rate": 0.02,
        #     # "architecture": "CNN",
        #     # "dataset": "CIFAR-100",
        #     # "epochs": 10,
        #     # }
        # )
        # wandb.watch(self.model, log="all")
        # self.wdb = wandb


    def F_norm_change(self):
        # ||A100-A20||_F / ||A20||_F
        a5 = (self.model.adp - self.a5).norm(p='fro') / self.a10.norm(p='fro')
        a10 = (self.model.adp - self.a10).norm(p='fro') / self.a10.norm(p='fro')
        a20 = (self.model.adp - self.a20).norm(p='fro') / self.a20.norm(p='fro')
        a30 = (self.model.adp - self.a30).norm(p='fro') / self.a30.norm(p='fro')
        a40 = (self.model.adp - self.a40).norm(p='fro') / self.a40.norm(p='fro')
        self.logger.info("F norm variation of adj: {:.4g}, {:.4g}, {:.4g}, {:.4g}, {:.4g}".format(a5, a10, a20, a30, a40))

    def on_epoch_end(self, epoch: int):
        super().on_epoch_end(epoch=epoch)

        if self.if_record_adj:
        
            self.a5 = self.a5.to(self.model.adp.device)
            self.a10 = self.a10.to(self.model.adp.device)
            self.a20 = self.a20.to(self.model.adp.device)
            self.a30 = self.a30.to(self.model.adp.device)
            self.a40 = self.a40.to(self.model.adp.device)

            if epoch == 5:
                self.a5 = self.model.adp.detach()

            if epoch == 10:
                self.a10 = self.model.adp.detach()

            if epoch == 20:
                self.a20 = self.model.adp.detach()

            if epoch == 30:
                self.a30 = self.model.adp.detach()

            if epoch == 40:
                self.a40 = self.model.adp.detach()

            self.F_norm_change()

    def count_flops(self, data, epoch, iter_num, train):
        """Count the number of flops in the model."""
        # flops = torchprofile.profile_macs(model, input)
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        macs = torchprofile.profile_macs(self.model, (history_data, future_data_4_dec, iter_num, epoch, train))
        flops = 2 * macs * self.iter_per_epoch

        self.logger.info("Number of flops per epoch: {:.4g} G".format(flops / 1e9))

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        if self.c_flops:
            self.count_flops(data=data, epoch=epoch, iter_num=iter_num, train=train)
            self.c_flops = False

        model_return = super().forward(data=data, epoch=epoch, iter_num=iter_num, train=train, **kwargs)
        
        return model_return 

   