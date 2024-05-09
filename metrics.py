from PIL import Image
import io
import os
import numpy as np

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch
from utils import compute_outputs

from PIL import Image
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

def insert_pixels(images, saliency_maps, baselines, percentiles):
    batch_size = images.shape[0]
    num_channels = images.shape[1]

    # Flatten the saliency maps and sort indices based on attribution scores
    flattened_indices = torch.argsort(saliency_maps.view(batch_size, -1), dim=1, descending=True)

    # Initialize a blank image for each image in the batch
    inserted_images = baselines.copy()

    return_items = []

    # Outer loop for inserting pixels
    for start_idx, end_index in zip(np.concatenate(([0],percentiles[1:]), percentiles)):
        # Get the current chunk of indices for each image in the batch
        indices_chunk = flattened_indices[:, start_idx : end_index]

        # Inner loop for inserting pixels for each image in the batch
        for batch_idx in range(batch_size):
            # Set the corresponding pixels in the inserted image to the original image
            for channel_idx in range(num_channels):
                inserted_images[batch_idx, channel_idx].view(-1)[indices_chunk[batch_idx]] = images[batch_idx, channel_idx].view(-1)[indices_chunk[batch_idx]]

        return_items.append(inserted_images.copy())

    return torch.stack(return_items)   #shape is (num_percentiles, batch_size, num_channels, H, W)

def insertion_score(saliency_maps, images, model, baseline, fraction=0.05):
    batch_size = images.shape[0]

    # Initialize a blank image for each image in the batch
    baselines = baseline.unsqueeze(1).repeat(batch_size, 1, 1, 1)

    # Range of total number of pixels to insert (5% of the total pixels)
    num_pixels_to_insert = np.arange(0, 1.05, 0.05)
    bokeh_images_list = insert_pixels(images, saliency_maps, baselines, num_pixels_to_insert)
    # Initialize prediction values

    prediction_values = compute_outputs(bokeh_images_list)

    # Convert the list of prediction values to a tensor
    prediction_values = torch.tensor(prediction_values).T

    # Compute the normalized curve
    normalized_curve = prediction_values / prediction_values[:, -1]

    # Compute the area under the curve (AUC)
    auc = torch.trapz(normalized_curve)

    return auc.item()

# Copyright 2022 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Implementation of Performance Information Curve metrics.

This module implements the saliency evaluation metrics as described in
"XRAI: Better Attributions Through Regions" (1). Use the compute_pic_metric(...)
method for computing both Softmax Information Curve (SIC) and Accuracy
Information Curve (AIC).

Here are the typical steps to compute the aggregated PIC curve for saliency
methods:
1) Call generate_random_mask(...) to generate a random pixel mask that serves
     as the seed for generating initial fully blurred image. To avoid using the
     same random mask for all images, generate a new mask for every new image.
     For fair comparison of multiple saliency methods, use the same random mask
     on a given image. If exact reproducibility is required, save the random mask
     on disk along with the results and reuse the masks on subsequent runs.
     The result of generate_random_mask(...) should be passed as the `random_mask`
     argument to compute_pic_metric(...).
2) Call compute_pic_metric(...) for every image and store the results in a list.
     Use separate lists for different saliency methods. Be aware, that the
     method can raise ComputePicMetricError. If that happens, skip the image.
     At the end of this step, you should have multiple lists with results: one
     list per saliency method.
3) Aggregate the results obtained in step 2 by calling
     aggregate_individual_pic_results(...). One call should be made per saliency
     method (i.e. per list obtained in step 2).

See the complementary pic_metrics.ipynb Jupyter notebook for an example of
usage.

(1) https://arxiv.org/abs/1906.02825
"""

import io
import numpy as np
import os

from PIL import Image
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

def estimate_image_entropy(image: np.ndarray) -> float:
    """Estimates the amount of information in a given image.

        Args:
            image: an image, which entropy should be estimated. The dimensions of the
                array should be [H, W, C] or [H, W] of type uint8.
        Returns:
            The estimated amount of information in the image.
    """
    buffer = io.BytesIO()
    pil_image = Image.fromarray(image)
    pil_image.save(buffer, format='webp', lossless=True, quality=100)
    buffer.seek(0, os.SEEK_END)
    length = buffer.tell()
    buffer.close()
    return length

def estimate_image_MSSSIM(bokehs: torch.Tensor, blurred: torch.Tensor) -> float:

    return ms_ssim( bokehs, blurred, data_range=255, size_average=False ) #(N,)

class ComputePicMetricError(Exception):
    """An error that can be raised by the compute_pic_metric(...) method.

    See the method description for more information.
    """
    pass


class PicMetricResult(NamedTuple):
    """Holds results of compute_pic_metric(...) method."""
    # x-axis coordinates of PIC curve data points.
    curve_x: Sequence[float]
    # y-axis coordinates of PIC curve data points.
    curve_y: Sequence[float]
    # A sequence of intermediate blurred images used for PIC computation with
    # the fully blurred image in front and the original image at the end.
    blurred_images: Sequence[np.ndarray]
    # Model predictions for images in the `blurred_images` sequence.
    predictions: Sequence[float]
    # Saliency thresholds that were used to generate corresponding
    # `blurred_images`.
    thresholds: Sequence[float]
    # Area under the curve.
    auc: float


def compute_pic_metric(
        img: np.ndarray,
        pred_probas: np.ndarray,
        percentiles: np.ndarray,
        bokeh_images: np.ndarray,
        method="compression",
        min_pred_value: float = 0.8,
        keep_monotonous: bool = True,
        num_data_points: int = 1000
) -> PicMetricResult:
    """Computes Performance Information Curve for a single image.

        The method can be used to compute either Softmax Information Curve (SIC) or
        Accuracy Information Curve (AIC). The method computes the curve for a single
        image and saliency map. This method should be called repeatedly on different
        images and saliency maps.

        Args:
            img: an original image on which the curve should be computed. The numpy
                array should have dimensions [H, W, C] for a color image or [H, W]
                for a grayscale image. The array should be of type uint8.
            pred_probas: prediction value for each bokeh image
            percentiles: a list of percentiles that should be used to generate the bokeh [0, 1] inclusive
            bokeh_images: the list of bokeh images generated by insert_pixels
            path: path to the blurred image.
            method: How to compress the image. The valid values are 'compression' and 'msssim'.
            min_pred_value: used for filtering images. If the model prediction on the
                original image is lower than the value of this argument, the method
                raises ComputePicMetricError to indicate that the image should be
                skipped. This is done to filter out images that produce low prediction
                confidence.
            keep_monotonous: whether to keep the curve monotonically increasing.
                The value of this argument was set to 'True' in the original paper but
                setting it to 'False' is a viable alternative.
            num_data_points: the number of PIC curve data points to return. The number
                excludes point 1.0 on the x-axis that is always appended to the end.
                E.g., value 1000 results in 1001 points evently distributed on the
                x-axis from 0.0 to 1.0 with 0.001 increment.

        Returns:
            The PIC curve data points and extra auxiliary information. See
            `PicMetricResult` for more information.

        Raises:
            ComputePicMetricError:
                The method raises the error in two cases. That happens in two cases:
                1. If the model prediction on the original image is not higher than the
                     model prediction on the completely blurred image.
                2. If the entropy of the original image is not higher than the entropy
                     of the completely blurred image.
                3. If the model prediction on the original image is lower than
                     `min_pred_value`.
                If the error is raised, skip the image.
    """
    if img.dtype.type != np.uint8:
        raise ValueError('The `img` array that holds the input image should be of'
                          ' type uint8. The actual type is {}.'.format(img.dtype))
    predictions = []

    # This list will contain mapping of image entropy for a given saliency
    # threshold to model prediction.
    entropy_pred_tuples = []

    # Estimate entropy of the original image.
    original_img_entropy = estimate_image_entropy(img)

    # Estimate entropy of the completely blurred image.
    fully_blurred_img = bokeh_images[0]
    
    if method == "compression":
        fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)

    # Compute model prediction for the original image.
    original_img_pred = pred_probas[-1]

    if original_img_pred < min_pred_value:
        message = ('The model prediction score on the original image is lower than'
                             ' `min_pred_value`. Skip this image or decrease the'
                             ' value of `min_pred_value` argument. min_pred_value'
                             ' = {}, the image prediction'
                             ' = {}.'.format(min_pred_value, original_img_pred))
        raise ComputePicMetricError(message)

    # If the score of the model on completely blurred image is higher or equal to
    # the score of the model on the original image then the metric cannot be used
    # for this image. Don't include this image in the aggregated result.
    if fully_blurred_img_pred >= original_img_pred:
        message = (
                'The model prediction score on the completely blurred image is not'
                ' lower than the score on the original image. Catch the error and'
                ' exclude this image from the evaluation. Blurred score: {}, original'
                ' score {}'.format(fully_blurred_img_pred, original_img_pred))
        raise ComputePicMetricError(message)
    
    # Compute model prediction for the completely blurred image.
    fully_blurred_img_pred = pred_probas[0]

    # If the entropy of the completely blurred image is higher or equal to the
    # entropy of the original image then the metric cannot be used for this
    # image. Don't include this image in the aggregated result.

    if fully_blurred_img_entropy >= original_img_entropy and method=="compression":
        message = (
                'The entropy in the completely blurred image is not lower than'
                ' the entropy in the original image. Catch the error and exclude this'
                ' image from evaluation. Blurred entropy: {}, original'
                ' entropy {}'.format(fully_blurred_img_entropy, original_img_entropy))
        raise ComputePicMetricError(message)

    # Iterate through saliency thresholds and compute prediction of the model
    # for the corresponding blurred images with the saliency pixels revealed.

    if method == "msssim":
        normalized_mssims = estimate_image_MSSSIM(bokeh_images, img[None, ...].repeat(len(bokeh_images), axis=0))
    max_normalized_pred = 0.0
    for i, (bokeh_image, pred) in enumerate(zip(bokeh_images, pred_probas)):
        if method=="compression":
            entropy = estimate_image_entropy(bokeh_image)
            # Normalize the values, so they lie in [0, 1] interval.
            normalized_entropy = (entropy - fully_blurred_img_entropy) / (
                    original_img_entropy - fully_blurred_img_entropy)
            
            normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
        else:
            normalized_entropy = normalized_mssims[i]

        normalized_pred = (pred - fully_blurred_img_pred) / (
                           original_img_pred - fully_blurred_img_pred)
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        max_normalized_pred = max(max_normalized_pred, normalized_pred)

        # Make normalized_pred only grow if keep_monotonous is true.
        if keep_monotonous:
            entropy_pred_tuples.append((normalized_entropy, max_normalized_pred))
        else:
            entropy_pred_tuples.append((normalized_entropy, normalized_pred))

    predictions = pred_probas.tolist()

    # Interpolate the PIC curve.
    entropy_pred_tuples.append((0.0, 0.0))

    entropy_data, pred_data = zip(*entropy_pred_tuples)
    interp_func = interpolate.interp1d(x=entropy_data, y=pred_data)

    curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points, endpoint=False)
    curve_y = np.asarray([interp_func(x) for x in curve_x])

    curve_x = np.append(curve_x, 1.0)
    curve_y = np.append(curve_y, 1.0)

    auc = np.trapz(curve_y, curve_x)

    return PicMetricResult(curve_x=curve_x, curve_y=curve_y,
                           blurred_images=bokeh_images,
                           predictions=predictions, thresholds=percentiles,
                           auc=auc)


class AggregateMetricResult(NamedTuple):
    """Holds results of aggregate_individual_pic_results(...) method."""
    # x-axis coordinates of aggregated PIC curve data points.
    curve_x: Sequence[float]
    # y-axis coordinates of aggregated PIC curve data points.
    curve_y: Sequence[float]
    # Area under the curve.
    auc: float


def aggregate_individual_pic_results(
        compute_pic_metrics_results: List[PicMetricResult],
        method: str = 'median') -> AggregateMetricResult:
    """Aggregates PIC metrics of individual images to produce the aggregate curve.

        The method should be called after calling the compute_pic_metric(...) method
        on multiple images for a given single saliency method.

        Args:
            compute_pic_metrics_results: a list of PicMetricResult instances that are
                obtained by calling compute_pic_metric(...) on multiple images.
            method: method to use for the aggregation. The valid values are 'mean' and
                'median'.
        Returns:
            AggregateMetricResult - a tuple with x, y coordinates of the curve along
                with the AUC value.


    """
    if not compute_pic_metrics_results:
        raise ValueError('The list of results should have at least one element.')

    curve_ys = [r.curve_y for r in compute_pic_metrics_results]
    curve_ys = np.asarray(curve_ys)

    # Validate that x-axis points for all individual results are the same.
    curve_xs = [r.curve_x for r in compute_pic_metrics_results]
    curve_xs = np.asarray(curve_xs)
    _, counts = np.unique(curve_xs, axis=1, return_counts=True)
    if not np.all(counts == 1):
        raise ValueError('Individual results have different x-axis data points.')

    if method == 'mean':
        aggr_curve_y = np.mean(curve_ys, axis=0)
    elif method == 'median':
        aggr_curve_y = np.median(curve_ys, axis=0)
    else:
        raise ValueError('Unknown method {}.'.format(method))

    auc = np.trapz(aggr_curve_y, curve_xs[0])

    return AggregateMetricResult(curve_x=curve_xs[0], curve_y=aggr_curve_y,
                                                             auc=auc)