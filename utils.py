import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from PIL import Image
from torchvision import models, transforms
import timm
import gc

cudnn.benchmark = True

def load_preprocess (name = "inceptionv3"):
    if(name == "resnet50v2" or name == "resnet101v2" or name == "resnet152v2" or name == "inceptionv3" or name == "mobilenetv2" or name == "vgg16" or name == "vgg19" or name == "densenet121" or name == "densenet169" or name == "densenet201"):
        def PreprocessImages(images):
            images = np.array(images)
            transformer = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            images = torch.tensor(images,dtype = torch.float32)
            images = transformer(images)
            return images.requires_grad_(True)
        return PreprocessImages
    if(name == "xception"):
        def PreprocessImages(images):
            images = np.array(images)
            transformer = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            images = torch.tensor(images, dtype=torch.float32)
            images = transformer(images)
            return images.requires_grad_(True)
        return PreprocessImages



def load_model(name = "inceptionv3"):
    model = None
    if(name == "resnet50v2"):
        model = models.resnet50(weights='IMAGENET1K_V2').cuda()
    if(name == "resnet101v2"):
        model = models.resnet101(weights='IMAGENET1K_V2').cuda()
    if(name == "resnet152v2"):
        model = models.resnet152(weights='IMAGENET1K_V2').cuda()
    if(name == "inceptionv3"):
        model = models.inception_v3(weights='IMAGENET1K_V1', init_weights=False).cuda()
    if(name == "mobilenetv2"):
        model = models.mobilenet_v2(weights='IMAGENET1K_V2').cuda()
    if(name == "vgg16"):
        model = models.vgg16(weights='IMAGENET1K_V1').cuda()
    if(name == "vgg19"):
        model = models.vgg19(weights='IMAGENET1K_V1').cuda()
    if(name == "densenet121"):
        model = models.densenet121(weights='IMAGENET1K_V1').cuda()
    if(name == "densenet169"):
        model = models.densenet169(weights='IMAGENET1K_V1').cuda()
    if(name == "densenet201"):
        model = models.densenet201(weights='IMAGENET1K_V1').cuda()
    if(name == "xception"):
        model = timm.create_model('xception', pretrained=True).cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.zero_grad()
    return model


def load_image_loader(name = "inceptionv3"):
    if(name == "resnet50v2" or name == "resnet101v2" or name == "resnet152v2" or name == "mobilenetv2"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(232),
            transforms.CenterCrop(224)
        ])
    if(name == "inceptionv3"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299)
        ])
    if(name == "vgg16" or name == "vgg19" or name == "densenet121" or name == "densenet169" or name == "densenet201"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

    if(name == "xception"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299)
        ])
    def LoadImages(file_paths):
        return np.asarray([transformer(Image.open(file_path).convert('RGB')) for file_path in file_paths])
    return LoadImages

def compute_outputs(self, input_tensor, indices, batchSize=64):
    original_shape = input_tensor.shape
    input_tensor = input_tensor.view(-1, input_tensor.shape[-3:])

    outputs = torch.zeros(input_tensor.shape[0])

    for i in range(0, input_tensor.shape[0], batchSize):
        # Get the current batch
        batch = input_tensor[i:i+batchSize].to('cuda:0', non_blocking=True)
        current_batchSize = len(batch)
        with torch.no_grad():
            output = torch.nn.Softmax(dim=1)(self.model(batch))

        # Select the outputs at the given indices
        output = output[torch.arange(output.shape[0]), indices[i:i+current_batchSize]]

        # Compute the gradients of the selected outputs with respect to the input
        outputs[i:i+current_batchSize] = output.detach().to('cpu:0', non_blocking=True)

        del output, batch
        torch.cuda.empty_cache()
        gc.collect()
    input_tensor.requires_grad = False
    return outputs.detach().reshape(original_shape[:-3])