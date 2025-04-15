from torchvision import transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, mode='train'):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.6),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'train':
            return [self.transform(x), self.transform(x)]
        else:
            return self.transform(x)
