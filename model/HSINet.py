import torch
import torch.nn as nn
import torch.nn.functional as F


class spatial_att_new(nn.Module):

    def __init__(self, input_channel):
        super(spatial_att_new,self).__init__()
        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_sa = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, bias=True)
        self.conv_1 = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, bias=True)
        self.conv_3 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.con_stride = nn.Conv2d(input_channel, input_channel, 3, stride=2, padding=1)

        self.con = nn.Conv2d(input_channel * 2, input_channel, 3, 1, 1)


    def forward(self, x):




        x0 = self.relu(self.bn(self.conv_3(self.con_stride(x))))        
        x1 = self.relu(self.bn(self.conv_1(x0)))     #(b,c,h/2,w/2)



        x2 = self.relu(self.bn(self.conv_3(self.con_stride(x1))))       
        x2 = self.relu(self.bn(self.conv_1(x2)))



        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)       

        x2 = self.conv_1(self.relu(self.bn(self.conv_1(self.conv_3(x2)))))
        x2_ = torch.sigmoid(x2)

        x1 = self.relu(self.bn(self.conv_3(x0)))
        x3 = x1 * x2_ * 2                                                                        

        x3 = torch.cat([x3, x2], dim=1)  #new
        x3 = self.relu(self.bn(self.con(x3)))



        x3 = self.relu(self.bn(self.conv_1(x3)))
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)   #(b,c,h,w)


        x3 = self.conv_3(x3)

        x3 = self.conv_1(self.relu(self.bn(self.conv_1(x3))))
        x4 = torch.sigmoid(x3)   

        if x4.shape == x.shape and x3.shape == x.shape:
            x4 = x4
            x3 = x3
        else:
            b, c, h, w = x.size()

            x4 = x4[:, :, :h, :w]
            x3 = x3[:, :, :h, :w]

        out = x * x4 * 2

        out = torch.cat([x3, out], dim=1) 
        out = self.relu(self.bn(self.con(out)))

        return out


class recursive_modle(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(recursive_modle, self).__init__()
        self.att = spatial_att_new(in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv_1 = nn.Conv2d(in_channels,in_channels, 1, 1)

    def forward(self, x):
        x = self.att(x)

        identity = self.conv_1(x)
        x1 = self.act(self.bn(self.conv_3(x)))
        x2 = self.conv_3(x1)

        x_out = x2 + identity
        out = self.act(self.bn(x_out))

        return out

class recon_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(recon_net, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = self.conv_1(x)
        x = self.act(self.bn(self.conv_3(x)))
        x = self.conv_3(x)
        out = identity + x
        out = self.act(self.bn(out))
        return out

class net_new(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels):
        super(net_new, self).__init__()
        self.conv_input = nn.Conv2d(in_channels, middle_channels, 3, 1, 1)
        self.recur_module = recursive_modle(middle_channels, middle_channels)
        self.recon = recon_net(in_channels, in_channels)

        self.conv_3 = nn.Conv2d(middle_channels, middle_channels, 3, 1, 1)
        self.conv_1 = nn.Conv2d(middle_channels, middle_channels, 1, 1)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.conv_out = nn.Conv2d(middle_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.act(self.bn(self.conv_input(x)))
        out = x
        out_all = 0
        for i in range(3):
            out = self.recur_module(out)
            # out = torch.add(out, x)
            identity1 = self.conv_1(out)
            x1 = self.act(self.bn(self.conv_3(out)))
            x1 = self.conv_3(x1)
            x1_out = identity1 + x1
            x1_out = self.act(self.bn(x1_out))
            out = self.recon(x1_out)
            out_all += out

        x2_out = out_all / 3


        # identity2 = self.conv_1(x1_out)
        # x2 = self.act(self.bn(self.conv_3(x1_out)))
        # x2 = self.conv_3(x2)
        # x2_out = identity2 + x2
        # x2_out = self.act(self.bn(x2_out))

        x = self.conv_out(x2_out)
        x = torch.sigmoid(x)
        return x




class hslnet(nn.Module):

    def __init__(self, in_channels, out_channels, middle_channels):
        super(hslnet, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels, middle_channels,  3, 1, 1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, 1, 1)

        self.att = spatial_att_new(middle_channels)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.conv3_1 = nn.Conv2d(middle_channels, middle_channels, 3, 1, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(middle_channels, middle_channels, 3, 1, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(middle_channels, middle_channels, 3, 1, padding=1, dilation=1)
        self.conv3_4 = nn.Conv2d(middle_channels, middle_channels, 3, 1, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(middle_channels, middle_channels, 1, 1)

        self.conv_y = nn.Conv2d(1,1, 3, 1, 1)  # 0804
        self.conv_z = nn.Conv2d(64, 64, 3, 1, 1)


    def forward(self, x):
        noise_input = x[:, 0:64, :, :]  # torch.Size([1, 32, 512, 512])
        y_input = x[:, 64:65, :, :]     # torch.Size([1, 1, 512, 512])  

        noise_input = self.conv_z(noise_input)
        # y = torch.sigmoid(self.conv_y(y_input))
        y = torch.sigmoid(self.conv_y(y_input))
        x = torch.cat([noise_input, y], dim=1)
        # x = noise_input + y
        # x = y * noise_input

        x = self.act(self.bn(self.conv_3(x)))


        identity1 = x
        x = self.act(self.bn(self.conv3_1(x)))
        x = self.conv3_2(x)
        x1 = x + self.conv1(identity1)
        x1 = self.act(self.bn(x1))



        identity2 = x1
        x = self.act(self.bn(self.conv3_3(x1)))
        x = self.conv3_4(x)
        x2 = x + self.conv1(identity2)
        x2 = self.act(self.bn(x2))

        identity3 = x2
        x = self.act(self.bn(self.conv3_3(x2)))
        x = self.conv3_2(x)
        x3 = x + self.conv1(identity3)
        x3 = self.act(self.bn(x3))


        x3 = self.att(x3)


        x = self.conv_1(x3)
        x = torch.sigmoid(x)
        # x = torch.clamp(x, 0, 1)
        return x

