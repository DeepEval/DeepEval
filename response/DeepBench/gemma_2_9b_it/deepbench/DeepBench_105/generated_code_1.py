import timm

def resnet50(weights=None, **kwargs):
  if weights is not None:
    input_channels = timm.create_model('resnet50', pretrained=False).default_cfg['input_channels']
    model = timm.create_model('resnet50', pretrained=False, **kwargs)
    model.load_state_dict(weights, strict=False)
    assert set(model.state_dict().keys()) >= set(timm.create_model('resnet50', pretrained=False).state_dict().keys())
  else:
    model = timm.create_model('resnet50', **kwargs)
  return model

if __name__ == "__main__":
  sample_input = torch.randn(1, 3, 224, 224)
  model = resnet50()
  output = model(sample_input)
  print(output.shape)