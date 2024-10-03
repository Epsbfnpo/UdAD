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

    def evaluate(self, val_loader):
        self.net_genA.eval()
        correct = 0
        total = 0
        class_correct = [0] * 3
        class_total = [0] * 3
        with torch.no_grad():
            for data in val_loader:
                self.set_input(data)
                outputs = self.net_genA(self.inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += self.labels.size(0)
                correct += (predicted == self.labels).sum().item()
                for i in range(len(self.labels)):
                    label = self.labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        accuracy = 100 * correct / total
        print(f"Overall Accuracy: {accuracy:.2f}%")
        for i in range(3):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"Accuracy of class {i}: {class_acc:.2f}%")
            else:
                print(f"No samples for class {i}")
        self.net_genA.train()
        return accuracy
