"""Nets: Neural network architectures for federated learning experiments.

Provides mock implementations of common FL backbone networks.
Each class follows a consistent interface: __init__(num_classes, ...) and forward(x).
"""
import numpy as np


class LeNet5:
    """LeNet-5: classic CNN for MNIST digit recognition.
    2 conv layers + 3 fc layers. ~60K parameters.
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.name = "LeNet-5"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class ResNet18:
    """ResNet-18: residual network with 18 layers.
    Standard backbone for CIFAR-10/100 classification.
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.name = "ResNet-18"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class ResNet50:
    """ResNet-50: deeper residual network with bottleneck blocks.
    Used for more complex vision tasks and ReID.
    """
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        self.name = "ResNet-50"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class VGG16:
    """VGG-16: deep convolutional network with small 3x3 filters.
    Popular for transfer learning in ReID tasks.
    """
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        self.name = "VGG-16"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class MobileNetV2:
    """MobileNetV2: lightweight CNN with inverted residuals.
    Suitable for resource-constrained federated clients.
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.name = "MobileNetV2"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class PointBERT:
    """PointBERT: BERT-style transformer for 3D point cloud understanding.
    Used as frozen backbone in FedULIP for multimodal 3D tasks.
    """
    def __init__(self, num_classes=40):
        self.num_classes = num_classes
        self.name = "PointBERT"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


class TextCNN:
    """TextCNN: 1D CNN for text classification (AG News).
    Multiple kernel sizes for n-gram feature extraction.
    """
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.name = "TextCNN"

    def forward(self, x):
        return np.random.randn(x.shape[0], self.num_classes) * 0.1


# Registry of available networks per task type
NETWORK_REGISTRY = {
    "vision_classification": ["LeNet-5", "ResNet-18", "ResNet-50", "MobileNetV2"],
    "text_classification": ["TextCNN"],
    "reid": ["ResNet-50", "VGG-16"],
    "ulip3d": ["PointBERT"],
}

# Lookup table
NET_LOOKUP = {
    "lenet5": LeNet5,
    "resnet18": ResNet18,
    "resnet50": ResNet50,
    "vgg16": VGG16,
    "mobilenetv2": MobileNetV2,
    "pointbert": PointBERT,
    "textcnn": TextCNN,
}


def get_network(network_id, num_classes=10):
    """Instantiate a network by its ID."""
    net_cls = NET_LOOKUP.get(network_id)
    if net_cls is None:
        return LeNet5(num_classes)
    return net_cls(num_classes)


def get_networks_for_task(task_type):
    """Return list of (id, name) pairs for a task type."""
    names = NETWORK_REGISTRY.get(task_type, NETWORK_REGISTRY["vision_classification"])
    return [(n.lower().replace("-", ""), n) for n in names]
