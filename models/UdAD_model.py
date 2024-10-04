from models.base_model import BaseModel
from models.nets import init_net
from models.nets.gen_net import GenNet
import torch.nn as nn
import torch


class UdADModel(BaseModel):
    def __init__(self, opt):
        super(UdADModel, self).__init__(opt)
        self.model_names = ["genA"]
        self.loss_names = ['classification']

        if self.isTrain:
            self.net_genA = init_net(GenNet(opt.input_nc, opt.cnum, opt.output_nc, (181, 217, 181)), opt.init_type, opt.init_gain, opt.gpu_ids)
            self.criterion_classification = nn.CrossEntropyLoss()
            self.optimizer_G = torch.optim.Adam(self.net_genA.parameters(), lr=1e-4, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)
            self.train_loss = 0.0
            self.train_correct = 0
            self.train_total = 0
        else:
            self.net_genA = init_net(GenNet(opt.input_nc, opt.cnum, opt.output_nc), opt.init_type, opt.init_gain, opt.gpu_ids)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--output_nc2', type=int, default=3, help='# UdAD Cycle OUTPUT2')
        return parser

    def set_input(self, input):
        self.b0 = input['b0'].to(self.device).type(dtype=torch.float)
        self.dwis = input['dwis'].to(self.device).type(dtype=torch.float)
        self.inputs = torch.cat([self.b0, self.dwis], dim=1)
        self.labels = input['label'].to(self.device).type(dtype=torch.long)

    def forward(self):
        self.logits = self.net_genA(self.inputs)

    def backward_g(self):
        self.loss_classification = self.criterion_classification(self.logits, self.labels)
        self.loss_classification.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

        self.train_loss += self.loss_classification.item() * self.inputs.size(0)
        _, predicted = torch.max(self.logits.data, 1)
        self.train_total += self.labels.size(0)
        self.train_correct += (predicted == self.labels).sum().item()

    def reset_metrics(self):
        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0
        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

    def evaluate(self, val_loader):
        self.net_genA.eval()
        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0
        class_correct = [0] * 3
        class_total = [0] * 3
        with torch.no_grad():
            for data in val_loader:
                self.set_input(data)
                outputs = self.net_genA(self.inputs)
                loss = self.criterion_classification(outputs, self.labels)
                self.val_loss += loss.item() * self.inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                self.val_total += self.labels.size(0)
                self.val_correct += (predicted == self.labels).sum().item()
                for i in range(len(self.labels)):
                    label = self.labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        val_loss_avg = self.val_loss / self.val_total
        val_accuracy = 100 * self.val_correct / self.val_total
        print(f"Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        for i in range(3):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"Accuracy of class {i}: {class_acc:.2f}%")
            else:
                print(f"No samples for class {i}")
        self.net_genA.train()
        return val_loss_avg, val_accuracy

    def get_training_metrics(self):
        train_loss_avg = self.train_loss / self.train_total
        train_accuracy = 100 * self.train_correct / self.train_total
        return train_loss_avg, train_accuracy

