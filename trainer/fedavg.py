from src.trainer.base import *


class Client(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)

    def local_train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === test ===
        # self.meters['acc'].append(self.local_test())
        self.meters['loss'].append(sum(batch_loss)/len(batch_loss))


class Server(BaseServer):
    def aggregate(self):
        client_num = len(self.clients)
        p_tensors = []
        for _, client in enumerate(self.clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / client_num
        self.model.tensor_to_parameters(avg_tensor)

    def train(self):
        self.sample()

        for client in self.sampled_clients:
            client.clone_model(self)

        for client in self.sampled_clients:
            client.local_train()

        self.aggregate()


