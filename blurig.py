import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import gc
from torchvision import transforms
import torch
import math

cudnn.benchmark = True

class BlurIG:
    def __init__(self, model, preprocess, load):
        self.model = model
        self.load = load
        self.preprocess = preprocess
        self.blurrers = dict()

    def gaussian_blur(self, images, sigma):
        if sigma == 0:
            return images.cpu()

        size = float(min(2*torch.round(4*sigma)+1, 101))
        if sigma not in self.blurrers:
            self.blurrers[sigma] = transforms.GaussianBlur(size, float(sigma)).cuda()
        return self.blurrers[sigma].forward(images).cpu()
        
    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = torch.zeros_like(input_tensor)
        outputs = torch.zeros(input_tensor.shape[0])

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].cuda()
            batch.requires_grad = True
            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))

            # Select the outputs at the given indices
            output = output[torch.arange(output.shape[0]), indices[i:i+current_batchSize]]

            # Compute the gradients of the selected outputs with respect to the input
            gradients[i:i+current_batchSize] = torch.autograd.grad(output, batch, grad_outputs=torch.ones_like(output), retain_graph=True)[0].detach().cpu()
            outputs[i:i+current_batchSize] = output.detach().cpu()

            del output, batch
            torch.cuda.empty_cache()
            gc.collect()
        input_tensor.requires_grad = False
        return outputs.detach(), gradients.detach()
    
    def saliency(self, image_paths, prediction_class, steps=20, steps_at=None, batch_size=32, max_sigma = 50, grad_step=0.01, sqrt=False):
        processed_images = self.preprocess(torch.from_numpy(self.load(image_paths))).requires_grad_(False)
        if sqrt:
            sigmas = torch.Tensor([math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)])
        else:
            sigmas = torch.Tensor([float(i)*max_sigma/float(steps) for i in range(0, steps+1)])
        step_vector_diff = sigmas[1:] - sigmas[:-1]

        if steps_at is None:
            steps_at = [steps]

        sequence           = torch.zeros((len(processed_images), steps, *processed_images.shape[1:]))
        gaussian_gradients = torch.zeros((len(processed_images), steps, *processed_images.shape[1:]))

        processed_images = processed_images.cuda()

        for i in range(steps):
            x_step = self.gaussian_blur(processed_images, sigmas[i])
            gaussian_gradient = (self.gaussian_blur(processed_images, sigmas[i]+grad_step)-x_step)/grad_step

            gc.collect()
            torch.cuda.empty_cache()

            gaussian_gradients[:, i] = gaussian_gradient
            sequence[:, i] = x_step

        processed_images = processed_images.cpu()

        target_class_idx = np.repeat(prediction_class, steps)
        outputs, gradients = self.compute_outputs_gradients(sequence.view(-1, *processed_images.shape[1:]), target_class_idx, batchSize=batch_size)
        outputs = outputs.view(*sequence.shape[:2])
        gradients = gradients.view(*sequence.shape)


        out_ig = []
        out_idgi = []
        for n in steps_at:
            gradients_copy = gradients[:,::steps//n]
            outputs_copy = outputs[:,::steps//n]
            gaussian_gradients_copy = gaussian_gradients[:,::steps//n]
            step_vector_diff_copy = step_vector_diff.reshape(n, steps//n).sum(1).view(1,-1,1,1,1)

            d = outputs_copy[:, 1:] - outputs_copy[:, :-1]
            element_product = gradients_copy[:,:-1]**2
                               
            a = -(step_vector_diff_copy*gaussian_gradients_copy*gradients_copy).sum((1, 2)).numpy()
            b = -(element_product*d.view(len(processed_images),n-1,1,1,1)/element_product.sum((2,3,4)).view(len(processed_images),n-1,1,1,1)).sum((1,2)).numpy()

            out_ig.append(a)
            out_idgi.append(b)

        return out_ig, out_idgi