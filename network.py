import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class ResidualBlock(nn.Module):
    '''
    Class representing a Residual Block, with the option to also perform down- or up-sampling.
    '''
    def __init__(self, ch, sampling_type=None):
        '''
        - inputs
        ch: number of channels of input tensor
        sampling_type: to flag if down- or up-sampling is performed, or not at all. Takes values {'up', 'down', None}
        '''
        
        super(ResidualBlock, self).__init__()
        
        self.sampling_type = sampling_type
        
        # convolutional layer
        self.conv = nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=ch, 
                                             kernel_size=(3,3), stride=1, 
                                             padding='same', bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=ch))
        
        # layer sequence to apply if down-sampling
        if sampling_type == 'down':
            self.sampling = nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=ch*2, 
                                                    kernel_size=(3,3), stride=1, 
                                                    padding='same', bias=False),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.BatchNorm2d(num_features=ch*2))
        
        # layer sequence to apply if up-sampling
        if sampling_type == 'up':
            self.sampling = nn.Sequential(nn.ConvTranspose2d(in_channels=ch, out_channels=ch//2,
                                                             kernel_size=3, stride=2,
                                                             bias=False),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(num_features=ch//2))
            
        
        
    def forward(self, x):
        '''
        Function to perform the forward pass.
        
        - input
        x: input tensor, shape = [batch_size, ch, height, width]
        
        - returns
        x: output tensor
        if sampling_type = None, shape = [batch_size, ch, height, width]
        if sampling_type = 'down', shape = [batch_size, ch*2, height, width]
        if sampling_type = 'up', shape = [batch_size, ch/2, height, width]
        s (optional): output of encoder before down-sampling if scaling='down', shape = [batch_size, ch, height, width]
        '''

        # residual block
        residual = x
        x = self.conv(x)
        x = self.conv(x)
        x += residual
        
        # down-sampling, also return encoder output for long skip connection
        if self.sampling_type == 'down':
            skip = x
            x = self.sampling(x)
            return x, skip
        # up-sampling
        elif self.sampling_type == 'up':
            x = self.sampling(x)
            return x
        # no sampling
        else:
            return x

    

class Res_U_Net(nn.Module):
    '''
    UNet-like model using residual blocks rather than standard convolutions.
    '''
    def __init__(self, init_ch=64, out_ch=3, num_levels=4):
        '''
        - inputs
        init_ch: number of channels at output of initial convolution layer
        out_ch: number of channels at output of final convolution layer
        num_levels: number of encoders/decoders, excluding bottleneck
        '''

        super(Res_U_Net, self).__init__()
        
        # define initial layer (in_channels = 3 due to RGB images)
        self.initial_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=init_ch, 
                                                     kernel_size=(3,3), stride=1, 
                                                     padding='same', bias=False),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(num_features=init_ch))
        
        # define Module List of encoders + down-sampling
        self.encoders = nn.ModuleList([ResidualBlock(init_ch * (2**i), sampling_type='down') for i in range(num_levels)])
        
        # define bottleneck layer sequences
        self.bottleneck_1 = ResidualBlock(init_ch * (2**(num_levels)), sampling_type=None)
        self.bottleneck_2 = ResidualBlock(init_ch * (2**(num_levels)), sampling_type='up')
        
        # define Module List of encoders + up-sampling
        self.decoders = nn.ModuleList([ResidualBlock(init_ch * (2**i), sampling_type='up') if i !=0 
                                       else ResidualBlock(init_ch * (2**i), sampling_type=None) 
                                       for i in reversed(range(num_levels))])
        
    
        #define final layer
        self.final_layer = nn.Sequential(nn.Conv2d(in_channels=init_ch, out_channels=out_ch, 
                                                   kernel_size=(1,1), stride=1, 
                                                   padding='same', bias=True),
                                         nn.Sigmoid())
        
        

    def forward(self, x):
        '''
        Function to perform forward pass of model.
        
        - input
        x: input tensor, shape = [batch_size, 3, height, width]
        
        -returns
        x: predicted segmentation mask, shape = [batch_size, self.out_ch, height, width]
        '''
            
        skips = [] # store skip layers
            
        # first layer
        x = self.initial_layer(x)
            
        # encoders: store output for skip connection then downsample
        for enc in self.encoders:
            x, s = enc(x)
            skips += [s]
                
        # bottleneck
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
            
        # decoders: resize output of up-sampling then add to skip connection, then up-sample
        for s, dec in zip(reversed(skips), self.decoders):
            x = CenterCrop(size=(s.shape[2],s.shape[3]))(x)
            x = dec(x)
            
        # final layer
        x = self.final_layer(x)
            
        return x