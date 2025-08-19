import torch.nn as nn
#Refine pose with TCN
class TCN(nn.Module): #Subclass of torch.nn.Module
  def __init__(self, num_joints=33, input_dim=3, hidden_dim=64):
        super().__init__()

        self.num_joints = num_joints
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        self.norm = nn.LayerNorm([num_joints, input_dim])  # Normalize per joint

  def forward(self, x):
        # x shape: (batch, frames, joints, 3)
        B, T, J, C = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * J, C, T)  # (B*J, C, T)

        y = self.conv1(x_reshaped)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.conv3(y)  # (B*J, C, T)

        y = y.reshape(B, J, C, T).permute(0, 3, 1, 2)  # (B, T, J, C)

        out = self.norm(x + y)  # Residual + normalization
        return out
