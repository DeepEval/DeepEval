from transformers import SwinTransformerV2

def swin_v2_b(pretrained=None, **kwargs):
  model = SwinTransformerV2("base", pretrained=pretrained, **kwargs)
  return model

if __name__ == "__main__":
  model = swin_v2_b(pretrained="facebook/swin-transformer-v2-base-patch16-224")
  sample_input = torch.randn(1, 3, 224, 224)
  output = model(sample_input)
  print(output.shape)