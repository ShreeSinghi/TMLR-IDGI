import numpy as np
import gc
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import gc
from tqdm import tqdm
import io
cudnn.benchmark = True

class IntegratedGradients:
    def __init__(self, model, preprocess, load):
        self.model = model
        self.preprocess = preprocess
        self.load = load

        # dimensions dont matter since it will be resized anyway
        BLACK  = Image.fromarray(np.zeros((299, 299, 3), dtype=np.uint8))
        bytes = io.BytesIO()
        BLACK.save(bytes,format="PNG")
        BLACK = preprocess(load([bytes]))
        self.BLACK = torch.Tensor(BLACK) # adds extra first dimension 


    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = torch.zeros_like(input_tensor)
        outputs = torch.zeros(input_tensor.shape[0])

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].to('cuda:0', non_blocking=True)
            batch.requires_grad = True
            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))

            # Select the outputs at the given indices
            output = output[torch.arange(output.shape[0]), indices[i:i+current_batchSize]]

            # Compute the gradients of the selected outputs with respect to the input
            gradients[i:i+current_batchSize] = torch.autograd.grad(output, batch, grad_outputs=torch.ones_like(output), retain_graph=True)[0].detach().to('cpu:0', non_blocking=True)
            outputs[i:i+current_batchSize] = output.detach().to('cpu:0', non_blocking=True)

            del output, batch
            torch.cuda.empty_cache()
            gc.collect()
        input_tensor.requires_grad = False
        return outputs.detach(), gradients.detach()

    def straight_path_images(self, images, n_steps):        
        x_diff = images - self.BLACK
        path_images = []
        
        for alpha in np.linspace(0, 1, n_steps):
            x_step = self.BLACK + alpha * x_diff
            path_images.append(x_step)
        
        path_images = torch.stack(path_images).transpose(0, 1)

        # returns x sequence
        return path_images

    def saliency(self, images, class_idxs, n_steps, compute_at=None,compute_batchSize=128):

        if compute_at is None:
            compute_at = [n_steps]

        sequence = self.straight_path_images(self.preprocess(self.load(images)), n_steps)

        image_shape = sequence.shape[2:]
        batchSize = sequence.shape[0]

        classes = np.repeat(class_idxs, n_steps)
        
        reshaped_sequence = sequence.reshape((batchSize*n_steps,)+image_shape).detach()

        output, gradients = self.compute_outputs_gradients(reshaped_sequence, classes, batchSize=compute_batchSize)
        gradients = gradients.view((batchSize, n_steps)+image_shape)
        output = output.view((batchSize, n_steps))
        sequence = sequence.detach()

        out_ig = []
        out_idgi = []
        for n in compute_at:
            gradients_copy = gradients[:,::n_steps//n]
            sequence_copy = sequence[:,::n_steps//n]
            outputs_copy = output[:,::n_steps//n]

            d = outputs_copy[:, 1:] - outputs_copy[:, :-1]
            element_product = gradients_copy[:,:-1]**2

            a = (gradients_copy[:,1:] * (sequence_copy[:,1:]-sequence_copy[:,:-1])).sum(1).abs().sum(1).numpy()
            b = (element_product*d.view(batchSize,n-1,1,1,1)/element_product.sum((2,3,4)).view(batchSize,n-1,1,1,1)).sum(1).abs().sum(1).numpy()

            out_ig.append(a/a.sum((1,2), keepdims=True))
            out_idgi.append(b/b.sum((1,2), keepdims=True))

        del reshaped_sequence, output, gradients, d, element_product, sequence
        torch.cuda.empty_cache()
        gc.collect()

        return out_ig, out_idgi

