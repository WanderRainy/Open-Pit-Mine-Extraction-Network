import torch
import torch.nn as nn

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, num_classes=1, evalmode=False):
        self.net = net(num_classes=num_classes).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.net.train()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss(num_classes=num_classes)
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch.cuda()
        if isinstance(mask_batch, dict):
            self.mask = {'label': mask_batch['label'].cuda(),
                         'height_label': mask_batch['height_label'].cuda()} 
        else:
            self.mask = mask_batch.cuda() if mask_batch is not None else None
        self.img_id = img_id

    def forward(self):
        if self.mask is not None:
            return self.net(self.img), self.mask
        else:
            return self.net(self.img)

    def optimize(self):
        self.optimizer.zero_grad()
        # import pdb
        # pdb.set_trace()
        pred = self.net(self.img)
        
        
        loss = self.loss(pred, self.mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f'update learning rate: {self.old_lr} -> {new_lr}', file=mylog)
        print(f'update learning rate: {self.old_lr} -> {new_lr}')
        self.old_lr = new_lr
