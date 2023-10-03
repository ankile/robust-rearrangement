import torch
import transformers


class DinoEncoder(torch.nn.Module):
    def __init__(self, size="base", freeze=True, device="cuda"):
        super().__init__()
        assert size in ["small", "base", "large", "giant"]
        self.device = device

        model_name = f"facebook/dinov2-{size}"
        self.trans = transformers.AutoImageProcessor.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name).to(self.device)
        self.encoding_dim = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

    def forward(self, x):
        x = self.trans(x, return_tensors="pt").pixel_values.to(self.device)
        x = self.model(x).pooler_output
        return x
