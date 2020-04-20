import torch.nn as nn
import torch

from blocks import attention_block, conv_block, ResidualBox


class MDADN(nn.Module):
    def __init__(self, num_object):
        super(MDADN, self).__init__()
        self.block_1 = nn.Sequential(*attention_block(5, 32))
        self.block_2 = nn.Sequential(*attention_block(32, 64))
        self.block_3 = nn.Sequential(ResidualBox(64, 128, [64]))
        self.block_4 = nn.Sequential(ResidualBox(128, 256, [128]))
        self.block_5 = nn.Sequential(ResidualBox(256, 512, [256, 512, 256]))
        self.block_6 = nn.Sequential(ResidualBox(512, 1024, [512, 1024, 512, 1024, 1024]))
        self.block_7 = nn.Sequential(*conv_block(1024 + 64, 1024))
        self.block_8 = nn.Conv2d(1024, num_object, 3, padding=1)
        self.skip_connection = nn.Sequential(*conv_block(256, 64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        skip = self.skip_connection(x)

        x = self.block_5(x)
        x = self.block_6(x)
        x = torch.cat([x, skip], dim=1)

        x = self.block_7(x)
        x = self.block_8(x)

        return x


if __name__ == '__main__':
    with torch.no_grad():
        model = MDADN(3).cuda()
        for i in range(100):
            image = torch.ones([1, 5, 224, 224]).cuda()
            image = model(image)
            print(image.shape)