"""
Siamese Network for Safaitic Glyph Similarity Learning.
Uses ResNet18 as backbone to learn embeddings for glyph comparison.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Custom L2Norm layer (since it's not in standard PyTorch)
class L2Norm(nn.Module):
    """L2 normalization layer."""
    
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)


class SafaiticSiameseNet(nn.Module):
    """
    Siamese Network for learning Safaitic glyph embeddings.
    Uses ResNet18 as shared backbone to extract 512-dimensional embeddings.
    """
    
    def __init__(self, embedding_dim=512):
        """
        Initialize the Siamese Network.
        
        Args:
            embedding_dim: Dimension of the output embedding (default: 512)
        """
        super(SafaiticSiameseNet, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        # ResNet18's avgpool + fc produces 1000-dim output
        # We'll replace fc with our own to get embedding_dim
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final fc layer
        
        # Get the number of features from ResNet18 (512 for resnet18)
        num_features = resnet.fc.in_features
        
        # Add our own embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            L2Norm(dim=1)  # Normalize embeddings to unit sphere
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Extract features using ResNet18 backbone
        features = self.backbone(x)
        
        # Get embedding
        embedding = self.embedding(features)
        
        return embedding
    
    def get_embedding(self, x):
        """
        Get embedding for a single input (useful for inference).
        
        Args:
            x: Input tensor of shape (1, 3, H, W) or (3, H, W)
        
        Returns:
            Embedding tensor
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
        
        return embedding


def euclidean_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding tensor (batch_size, embedding_dim)
        embedding2: Second embedding tensor (batch_size, embedding_dim)
    
    Returns:
        Distance tensor of shape (batch_size,)
    """
    return torch.norm(embedding1 - embedding2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    Minimizes distance for positive pairs, maximizes for negative pairs.
    """
    
    def __init__(self, margin=2.0):
        """
        Initialize Contrastive Loss.
        
        Args:
            margin: Margin for negative pairs (default: 2.0)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Binary label (1 for positive pairs, 0 for negative pairs)
                  Shape: (batch_size,)
        
        Returns:
            Loss value
        """
        # Calculate Euclidean distance
        distance = euclidean_distance(embedding1, embedding2)
        
        # Convert label to float
        label = label.float()
        
        # Positive pairs: minimize distance
        # Negative pairs: maximize distance (but with margin)
        positive_loss = label * torch.pow(distance, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        # Total loss
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss

