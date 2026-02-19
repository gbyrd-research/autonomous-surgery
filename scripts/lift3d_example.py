import torch
from autonomous_surgery.models.autonomous_surgery.model_loader import autonomous_surgery_clip_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = autonomous_surgery_clip_base().to(device)
point_cloud = torch.randn([4, 1024, 3]).to(device)
output = model(point_cloud)

print(output.shape)
