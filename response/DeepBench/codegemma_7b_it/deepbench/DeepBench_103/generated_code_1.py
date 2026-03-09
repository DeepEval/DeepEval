from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k

def swin_v2_b(pretrained=False, **kwargs):
    model = swin_base_patch4_window7_224_in22k(pretrained=pretrained, **kwargs)
    return model

if __name__ == "__main__":
    sample_input = torch.randn(2, 3, 224, 224)
    model = swin_v2_b(pretrained=True)
    output = model(sample_input)
    print(output.shape)