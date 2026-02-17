# import torch
# from torchvision.transforms import v2
# from pathlib import Path

# current_dir = Path(__file__).parent.resolve()
# DINO_REPO_DIR = current_dir.joinpath("..", "dinov3")
# DINO_MODEL_CKPT_URL = "https://dl.fbaipublicfiles.com/dinov3/dinov3_vit7b16.pth"

# def make_transform(resize_size: int = 256):
#     to_tensor = v2.ToImage()
#     resize = v2.Resize((resize_size, resize_size), antialias=True)
#     to_float = v2.ToDtype(torch.float32, scale=True)
#     normalize = v2.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )
#     return v2.Compose([to_tensor, resize, to_float, normalize])

# def load_dinov3(model_name: str):
#     return torch.hub.load(DINO_REPO_DIR, model_name, source='local', weights=DINO_MODEL_CKPT_URL)

if __name__=="__main__":
    
    from transformers import pipeline
    from transformers.image_utils import load_image

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = load_image(url)

    feature_extractor = pipeline(
        model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        task="image-feature-extraction", 
    )
    features = feature_extractor(image)