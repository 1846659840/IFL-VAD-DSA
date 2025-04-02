import torch
import torch.nn as nn

class ImageBindTransformerExtractor(nn.Module):
    def __init__(self, feature_dim=768, num_frames=16):
        super(ImageBindTransformerExtractor, self).__init__()
        # Set number of frames
        self.num_frames = num_frames

        # Load pre-trained ImageBind model
        try:
            from imagebind import imagebind_model
            from imagebind.models.imagebind_model import ImageBindModel

            # Initialize ImageBind model
            self.imagebind = imagebind_model.imagebind_huge(pretrained=True)

            # Freeze ImageBind feature extractor parameters
            for param in self.imagebind.parameters():
                param.requires_grad = False

        except ImportError:
            print("ImageBind not available, using ResNet as alternative feature extractor")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
            self.imagebind = nn.Sequential(*modules)
            feature_dim = 2048  # ResNet50 output dimension

        # Transformer encoder module - process frame sequences
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=2
        )

        # Output mapping layer
        self.output_layer = nn.Linear(feature_dim, 128)

    def forward(self, x):
        """
        Process 16-frame video sequence
        x: input shape [batch_size, num_frames, channels, height, width]
           or for single frame input [batch_size, channels, height, width]
        """
        # Check if input is multi-frame
        if len(x.shape) == 5:  # [batch_size, num_frames, channels, height, width]
            batch_size, num_frames, c, h, w = x.shape

            # Extract features for each frame
            frame_features = []

            for frame_idx in range(num_frames):
                # Extract single frame
                frame = x[:, frame_idx]  # [batch_size, channels, height, width]

                try:
                    with torch.no_grad():
                        # Process using ImageBind
                        image_inputs = {"vision": frame}
                        embeddings = self.imagebind(image_inputs)
                        frame_feat = embeddings["vision"]  # [batch_size, feature_dim]
                except (AttributeError, TypeError):
                    # Use alternative ResNet
                    frame_feat = self.imagebind(frame).view(batch_size, -1)

                frame_features.append(frame_feat)

            # Stack all frame features [num_frames, batch_size, feature_dim]
            features = torch.stack(frame_features, dim=0)

        else:  # Single frame input [batch_size, channels, height, width]
            try:
                with torch.no_grad():
                    image_inputs = {"vision": x}
                    embeddings = self.imagebind(image_inputs)
                    features = embeddings["vision"].unsqueeze(0)  # [1, batch_size, feature_dim]
            except (AttributeError, TypeError):
                batch_size = x.shape[0]
                features = self.imagebind(x).view(batch_size, -1).unsqueeze(0)  # [1, batch_size, feature_dim]

        # Process sequence through Transformer
        # features shape: [seq_len, batch_size, feature_dim]
        transformed = self.transformer(features)

        # If multi-frame input, average pool all frame features
        if len(x.shape) == 5:
            # [seq_len, batch_size, feature_dim] -> [batch_size, feature_dim]
            transformed = transformed.mean(dim=0)
        else:
            # Remove sequence dimension for single frame case
            transformed = transformed.squeeze(0)

        # Feature mapping
        output = self.output_layer(transformed)

        return output 