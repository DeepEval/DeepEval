import timm

def vit_small_patch16_224(pretrained=None, **kwargs):
  if pretrained:
    checkpoint = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    num_channels = checkpoint.head.in_features
    model = timm.create_model('vit_small_patch16_224', num_classes=1000, pretrained=False)
    model.head = timm.create_model('vit_small_patch16_224', num_classes=1000, pretrained=False).head
    model.load_state_dict(checkpoint.state_dict())
  else:
    model = timm.create_model('vit_small_patch16_224', pretrained=False)

  return model

if __name__ == "__main__":
    model = vit_small_patch16_224(pretrained='imagenet')
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)