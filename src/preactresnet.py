import torch
import torch.nn as nn
import torch.nn.functional as F




class PreActBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, bias=False):
    super(PreActBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    # スキップ結合
    if stride != 1 or in_channels != out_channels:
      # 出力の空間方向のサイズまたチャネル数が入力と異なる場合は、
      # スキップ結合に1x1畳み込みを噛ませてちゃんと足し算できるようにする。
      self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
    else:
      # こっちはただただ何もしないIdentity(恒等)関数。
      # PyTorchではよくIdentityをnn.Sequential()で実現する。
      self.shortcut = nn.Sequential()
  
  def forward(self, x):
    """bn -> relu -> conv -> bn -> relu -> conv -> add"""
    out = self.conv1(F.relu(self.bn1(x, inplace=True)))
    out = self.conv2(F.relu(self.bn2(out, inplace=True)))
    return out + self.shortcut(x)
  
class SEModule(nn.Module):
  """Squeeze-and-Excitation Module"""
  def __init__(self, in_channels, r=16):
    super(SEModule, self).__init__()
    h_feats = int(in_channels//r)
    # Squeeze
    self.gap = nn.AdaptiveAvgPool2d(1)
    # Excitation
    self.fc1 = nn.Linear(in_channels, h_feats)
    self.fc2 = nn.Linear(h_feats,in_channels)
  def forward(self, x):
    b,c,h,w = x.size()
    # Squeeze
    out = self.gap(x).view(b, -1)
    # Excitation
    out = F.relu(self.fc1(out))
    out = torch.sigmoid(self.fc2(out)).view(b,c,1,1)
    return x*out.expand_as(x)
  

class PreActSEBlock(nn.Module):
  """PreActBlock + SEModule"""
  def __init__(self, in_channels, out_channels, stride=1, bias=False,r=16):
    super(PreActSEBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    self.se_module = SEModule(out_channels)

    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
    else:
      self.shortcut = nn.Sequential()
  
  def forward(self, x):
    """bn -> relu -> conv -> bn -> relu -> conv -> SE -> add"""
    out = self.conv1(F.relu(self.bn1(x, inplace=True)))
    out = self.conv2(F.relu(self.bn2(out, inplace=True)))

    out = self.se_module(out)

    return out + self.shortcut(x)
  

swish = lambda x : x*torch.sigmoid(x)


class PreActSESwishBlock(nn.Module):
  """PreActBlock + SEModule + Swish"""
  def __init__(self, in_channels, out_channels, stride=1, bias=False,r=16):
    super(PreActSESwishBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    self.se_module = SEModule(out_channels)

    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
    else:
      self.shortcut = nn.Sequential()
  
  def forward(self, x):
    """bn -> swish -> conv -> bn -> swish -> conv -> SE -> add"""
    out = self.conv1(swish(self.bn1(x)))
    out = self.conv2(swish(self.bn2(out)))
    out = self.se_module(out)
    return out + self.shortcut(x)


class PreActResNet18(nn.Module):
  """PreAct-ResNet18"""
  def __init__(self,n_c=3, num_classes=10):
    super(PreActResNet18, self).__init__()
    # ステージ1
    self.first = nn.Sequential(
        nn.Conv2d(n_c, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
    )
    # ステージ2
    self.blc1 = PreActSESwishBlock(64, 64)
    self.blc2 = PreActSESwishBlock(64, 64)
    # ステージ3
    self.blc3 = PreActSESwishBlock(64, 128, stride=2)
    self.blc4 = PreActSESwishBlock(128, 128)
    # ステージ4
    self.blc5 = PreActSESwishBlock(128, 256, stride=2)
    self.blc6 = PreActSESwishBlock(256, 256)
    # ステージ5
    self.blc7 = PreActSESwishBlock(256, 512, stride=2)
    self.blc8 = PreActSESwishBlock(512, 512)

    # GAP -> 全結合層
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(512, num_classes)


  def forward(self,x):
    b, _, _, _ = x.size()
    out = swish(self.first(x))
    out = self.blc2(self.blc1(out))
    out = self.blc4(self.blc3(out))
    out = self.blc6(self.blc5(out))
    out = self.blc8(self.blc7(out))

    # GAP -> 全結合層
    out = self.pool(out).view(b, -1)
    out = self.fc(out)
    return out
  
class PreActResNet34(nn.Module):
  """PreAct-ResNet34"""
  def __init__(self,n_c=3, num_classes=10):
    super(PreActResNet34, self).__init__()
    # ステージ1
    self.first = nn.Sequential(
        nn.Conv2d(n_c, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
    )
    # ステージ2
    self.blc1 = PreActSESwishBlock(64, 64)
    self.blc2 = PreActSESwishBlock(64, 64)
    self.blc3 = PreActSESwishBlock(64, 64)
    # ステージ3
    self.blc4 = PreActSESwishBlock(64, 128, stride=2)
    self.blc5 = PreActSESwishBlock(128, 128)
    self.blc6 = PreActSESwishBlock(128, 128)
    self.blc7 = PreActSESwishBlock(128, 128)
    # ステージ4
    self.blc8 = PreActSESwishBlock(128, 256, stride=2)
    self.blc9 = PreActSESwishBlock(256, 256)
    self.blc10 = PreActSESwishBlock(256, 256)
    self.blc11 = PreActSESwishBlock(256, 256)
    self.blc12 = PreActSESwishBlock(256, 256)
    self.blc13 = PreActSESwishBlock(256, 256)
    # ステージ5
    self.blc14 = PreActSESwishBlock(256, 512, stride=2)
    self.blc15 = PreActSESwishBlock(512, 512)
    self.blc16 = PreActSESwishBlock(512, 512)

    # GAP -> 全結合層
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(512, num_classes)


  def forward(self,x):
    b, _, _, _ = x.size()
    out = swish(self.first(x))
    out = self.blc2(self.blc1(out))
    out = self.blc4(self.blc3(out))
    out = self.blc6(self.blc5(out))
    out = self.blc8(self.blc7(out))
    out = self.blc10(self.blc9(out))
    out = self.blc12(self.blc11(out))
    out = self.blc14(self.blc13(out))
    out = self.blc16(self.blc15(out))

    # GAP -> 全結合層
    out = self.pool(out).view(b, -1)
    out = self.fc(out)
    return out