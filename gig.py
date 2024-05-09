import torch
import torch.backends.cudnn as cudnn
import math
import numpy as np
import gc
from tqdm import tqdm

cudnn.benchmark = True

epsilon = 1E-7

def l1_distance(x1, x2):
  return np.abs(x1 - x2).sum()

def translate_x_to_alpha(x, x_input, x_baseline):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(x_input - x_baseline != 0,
                    (x - x_baseline) / (x_input - x_baseline), np.nan)

def translate_alpha_to_x(alpha, x_input, x_baseline):
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha

class GIG:
    def __init__(self, model, preprocess, load):
        self.model = model
        self.preprocess = preprocess
        self.load = load

    def compute_outputs_gradients(self, model, input_tensor, indices, batchSize=64):
        input_tensor.requires_grad = True
        gradients = torch.zeros_like(input_tensor)
        outputs = torch.zeros(input_tensor.shape[0])

        for i in range(0, input_tensor.shape[0], batchSize):
            batch = input_tensor[i:i+batchSize]#.to('cuda:0', non_blocking=True)
            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch.cuda()))

            output = output[torch.arange(output.shape[0]), indices[i:i+current_batchSize]]

            gradients[i:i+current_batchSize] = torch.autograd.grad(output, batch, grad_outputs=torch.ones_like(output), retain_graph=True)[0].detach().to('cpu:0', non_blocking=True)
            outputs[i:i+current_batchSize] = output.detach()

            del output, batch
            torch.cuda.empty_cache()
            gc.collect()
        return outputs, gradients

    def saliency(self, x_paths, class_idxs, steps=25, compute_at=[], batchSize=64, fraction=0.50, max_dist=1.0):

      x_orig = self.load(x_paths)
      x_input = x_orig.astype(np.float32)

      x_input_all = np.asarray(x_input, dtype=np.float64)
      x_baseline = np.zeros(x_input.shape)
      x_baseline = np.asarray(x_baseline, dtype=np.float64)
      x_all = x_baseline.copy()
      l1_total_all = np.asarray([l1_distance(x_input, x_baseline[0]) for x_input in x_input_all])
      attr_all = np.zeros_like(x_input, dtype=np.float64)
      result_all = np.zeros_like(x_input, dtype = np.float64)

      for step in range(steps):
        x_preprocessed_all = self.preprocess(x_all)
        outputs_all, grad_actual_all = self.compute_outputs_gradients(self.model, x_preprocessed_all, class_idxs, batchSize=batchSize)
        outputs_all, grad_actual_all = outputs_all.numpy(), grad_actual_all.numpy()
        out_all = outputs_all.copy()
        grad_all = grad_actual_all.copy()
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)

        for i in range(len(x_all)):
          gamma = np.inf
          x = x_all[i]
          x_input = x_input_all[i]
          grad_actual = grad_actual_all[i]
          out = outputs_all[i]
          grad = grad_all[i]
          x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline[0])
          x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline[0])
          l1_target = l1_total_all[i] * (1 - (step + 1) / steps)
          result = result_all[i]

          if step != 0:
            d = out - output_s[i]
            element_product = grad_s[i]**2
            result += element_product*d/element_product.sum()

          while gamma > 1.0:
            x_old = x.copy()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline[0])
            x_alpha[np.isnan(x_alpha)] = alpha_max
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            l1_current = l1_distance(x, x_input)
            if math.isclose(l1_target, l1_current, rel_tol=epsilon, abs_tol=epsilon):
              attr_all[i] += (x - x_old) * grad_actual
              break

            grad[x == x_max] = np.inf

            threshold = np.quantile(np.abs(grad), fraction, method='lower')
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

            l1_s = (np.abs(x - x_max) * s).sum()
            if l1_s > 0:
              gamma = (l1_current - l1_target) / l1_s
            else:
              gamma = np.inf

            if gamma > 1.0:
              x[s] = x_max[s]
            else:
              assert gamma > 0, gamma
              x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            attr_all[i] += (x - x_old) * grad_actual

        output_s = out_all.copy()
        grad_s = grad_actual_all.copy()
	

      return [attr_all.sum(1)], [result_all.sum(1)]
