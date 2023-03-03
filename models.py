import torch
from torchvision.models import resnet18, resnet50


class HeadNet(torch.nn.Module):

    def __init__(self, num_classes, feature_dim, image_shape):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.image_shape = image_shape
        # 5次下采样(128*96)→(4*3)
        self.extract_feature = torch.nn.Linear(
            self.feature_dim * int(image_shape[0] / 32) * int(image_shape[1] / 32), self.feature_dim)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = torch.nn.Linear(self.feature_dim, num_classes)


class BackboneByResNet18(HeadNet):
    FEATURE_DIM = 512
    def __init__(self, num_classes, image_shape):
        super().__init__(num_classes, self.FEATURE_DIM, image_shape)
        self.base = resnet18(pretrained=True)
        # self.base = resnet18(pretrained=False)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None

        # 标准化，向量变为模长为1
        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        # feature_normed = feature

        return logits, feature_normed


class BackboneByResNet50(HeadNet):
    FEATURE_DIM = 2048
    def __init__(self, num_classes, image_shape):
        super().__init__(num_classes, self.FEATURE_DIM, image_shape)
        # self.base = resnet50(pretrained=True)
        self.base = resnet50(pretrained=False)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None

        # 标准化，向量变为模长为1
        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )

        # feature_normed = feature

        return logits, feature_normed


def create(name, num_classes, image_shape):
    if name == "res18":
        model = BackboneByResNet18(num_classes, image_shape)
    else:
        model = BackboneByResNet50(num_classes, image_shape)

    return model
