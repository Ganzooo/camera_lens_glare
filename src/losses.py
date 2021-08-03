import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms



def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class CharbEdgeLoss(nn.Module):
    """Charbonnier Loss (L1) + Edge Loss"""

    def __init__(self, eps=1e-3):
        super(CharbEdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.charb = CharbonnierLoss(eps=eps)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]       # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4          # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss_c = self.charb(x, y)
        loss_e = self.charb(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss_c + 0.05 * loss_e


class LossWithMask(nn.Module):

    def __init__(self, loss_type='mse', eps=1e-3):
        super(LossWithMask, self).__init__()
        self.loss_type = loss_type
        self.eps = eps

    def maskedMSELoss(self, pred, target, mask):
        diff2 = (torch.flatten(pred) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        mse = torch.sum(diff2) / (torch.sum(mask) + self.eps)
        return mse

    def forward(self, pred, target, input):
        # mask(set to 0) if all channels are great and equal 0.98 (flare area)
        mask = torch.zeros(input.shape[0], 1, input.shape[2], input.shape[3]).cuda()
        for i in range(input.shape[0]):
            mask[i,0,:,:] = torch.logical_and(input[i,0,:,:].ge(0.99), input[i,1,:,:].ge(0.93))
            mask[i,0,:,:] = ~torch.logical_and(mask[i,0,:,:], input[i,2:,:,:].ge(0.99))
            mask = mask.expand(-1, 3, -1, -1)
        loss_flare = self.maskedMSELoss(pred, target, 1-mask)
        loss_other = self.maskedMSELoss(pred, target, mask)
        return 0.6 * loss_flare + 0.4 * loss_other


class ContentLossWithMask(nn.Module):

    def __init__(self, loss_type='mse', eps=1e-3):
        super(ContentLossWithMask, self).__init__()
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
        self.eps = eps
        
        # vgg-19
        with torch.no_grad():
            self.model = self.vgg19partial()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def vgg19partial(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, pred, target, input):
        # mask(set to 0) if all channels are great and equal 0.98 (flare area)
        mask = torch.zeros(input.shape[0], 1, input.shape[2], input.shape[3]).cuda()
        for i in range(input.shape[0]):
            mask[i,0,:,:] = torch.logical_and(input[i,0,:,:].ge(0.99), input[i,1,:,:].ge(0.93))
            mask[i,0,:,:] = ~torch.logical_and(mask[i,0,:,:], input[i,2:,:,:].ge(0.99))
            mask = mask.expand(-1, 3, -1, -1)

        # Copy target masked image into pred one
        pred_c = target * (1-mask) + pred * (mask)

        # vgg19 inputs(pred/target) should be normalized as [0, 1]
        t_pred_c = pred_c.clone()
        t_target = target.clone()
        for i in range(input.shape[0]):
            t_pred_c[i,:,:,:] = self.transform(t_pred_c[i,:,:,:])
            t_target[i,:,:,:] = self.transform(t_target[i,:,:,:])
        f_pred_c = self.model.forward(t_pred_c)
        f_target = self.model.forward(t_target)

        # MSE loss + Content loss
        loss_m = self.criterion(pred_c, target)
        loss_c = self.criterion(f_pred_c, f_target.detach())
        return 0.5 * loss_m + 0.006 * loss_c


class H2GLossWithMask(nn.Module):
    """H2G Loss : once for all"""
    def __init__(self, loss_type='mse', eps=1e-3):
        super(H2GLossWithMask, self).__init__()
        if loss_type == 'mse':
            self.criterion_m = nn.MSELoss()
            self.criterion_c = nn.MSELoss()
        elif loss_type == 'charb':
            self.criterion_m = CharbonnierLoss()
            self.criterion_c = nn.L1Loss()
        elif loss_type == 'charb_edge':
            self.criterion_m = CharbEdgeLoss()
            self.criterion_c = nn.L1Loss()
        else:
            self.criterion_m = nn.L1Loss()
            self.criterion_c = nn.L1Loss()
        self.eps = eps

        # vgg-19
        with torch.no_grad():
            self.model = self.vgg19partial()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def vgg19partial(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, pred_t, target):
        # vgg19 inputs(pred/target) should be normalized as [0, 1]
        t_pred_t = pred_t.clone()
        t_target = target.clone()
        for i in range(target.shape[0]):
            t_pred_t[i,:,:,:] = self.transform(t_pred_t[i,:,:,:])
            t_target[i,:,:,:] = self.transform(t_target[i,:,:,:])
        f_pred_t = self.model.forward(t_pred_t)
        f_target = self.model.forward(t_target)

        # Main loss + Content loss
        loss_m = self.criterion_m(pred_t, target)
        loss_c = self.criterion_c(f_pred_t, f_target.detach())
        return 0.5 * loss_m + 0.006 * loss_c