import tensorflow as tf
from typing import List

# import kornia
import torch
import torch.nn.functional as F


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='floor')

    return tuple(reversed(out))


def stich_windows(windows, k, cropx, cropy):
    if not torch.is_tensor(windows):
        windows = torch.as_tensor(windows)
    row0 = torch.cat([windows[0, 0][:-k, :-k]] +
                     [win[:-k, k:-k] for win in windows[0, 1:-1]] +
                     [windows[0, -1][:-k, k:]], dim=1)
    rows = []

    for r in range(windows.shape[0] - 1):
        r = r + 1
        rows.append(torch.cat([windows[r, 0][k:-k, :-k]] +
                              [win[k:-k, k:-k] for win in windows[r, 1:-1]] +
                              [windows[r, -1][k:-k, k:]], dim=1)
                    )

    row_last = torch.cat([windows[-1, 0][k:, :-k]] +
                         [win[k:, k:-k] for win in windows[-1, 1:-1]] +
                         [windows[-1, -1][k:, k:]], dim=1)

    final = torch.cat([row0] + rows + [row_last], dim=0)
    final = final[:cropx, :cropy]
    return final


def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)

class LocatorTF:
    def __init__(self, fastrcnn_model, process_stride=64, method='max', dark_threshold=20, locating_model=None,
                 mode='static', **kwargs):
        self.fastrcnn_model = fastrcnn_model
        self.mode = mode
        self.process_stride = process_stride
        self.method = method
        self.locating_model = locating_model
        self.dark_threshold = dark_threshold
        self.p_list = kwargs.get('p_list', [8, 6, 1.5, 1, 50])
        self.meanADU = kwargs.get('meanADU', 241.0)
        self.dynamic_thres = kwargs.get('dynamic_thres', True)
        self.pretune_thresholding = kwargs.get('pretune_thresholding')




    def model_tune(self, arr):
        meanADU = self.meanADU * 4  # mean ADU * upsample_factor^2
        offset = 0
        limit = int(tf.reduce_sum(arr) / meanADU + offset)
        arr_t = tf.cast(arr[None, None, ...] > 30, tf.float32)
        
        # Assuming a custom implementation for connected components that returns a count
        # since TensorFlow doesn't have a direct equivalent.
        limit_cca = connected_components(arr_t, num_iterations=10)
        limit = max(limit_cca, limit)
        limit = max(limit, 1)

        self.fastrcnn_model.rpn._pre_nms_top_n = {'training': limit * self.p_list[0], 'testing': limit * self.p_list[0]}
        self.fastrcnn_model.rpn._post_nms_top_n = {'training': limit * self.p_list[1],
                                                   'testing': limit * self.p_list[1]}
        self.fastrcnn_model.roi_heads.detections_per_img = int(limit * self.p_list[2])
        # self.fastrcnn_model.roi_heads.score_thresh = self.p_list[3] / limit if limit < self.p_list[4] else 0
        self.fastrcnn_model.roi_heads.score_thresh = self.p_list[3] / limit

        self.fastrcnn_model.roi_heads.nms_thresh = 0.02  # smaller, delete more detections

        if limit > (0.005 * arr.shape[0] * arr.shape[1]) and self.dynamic_thres:  # 0.002 is minimum for model13
            self.dark_threshold = 0  # for image that not quite sparse, lift the pre-thresholding.




    def images_to_window_lists(self, inputs):
        outputs = []
        maxs = []
        mins = []
        h, w = inputs.shape[1], inputs.shape[2]

        # TensorFlow's way of defining the upsample layer
        upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')

        if self.process_stride is None:
            for image in inputs:
                image = tf.expand_dims(image, 0)  # Add batch dimension
                windows = upsample(image)
                outputs.extend(tf.reshape(windows, [-1, *windows.shape[2:]]))
                maxs.extend([tf.reduce_max(image)] * (windows.shape[1] * windows.shape[2]))
                mins.extend([tf.reduce_min(image)] * (windows.shape[1] * windows.shape[2]))
        else:
            # Asserts in TensorFlow work differently, and dynamic checks like this are less common,
            # but you can still perform a check using tf.debugging.assert_positive, for example.
            for image in inputs:
                # Pad the image if necessary. This section will need to be adapted based on how you wish to pad.
                # TensorFlow uses 'reflect' mode for similar behavior to your PyTorch code.
                # The calculation of 'pad' needs to be adapted to TensorFlow.
                pad_height = tf.math.floordiv(h, self.process_stride - 6) * (self.process_stride - 6) + self.process_stride
                pad_width = tf.math.floordiv(w, self.process_stride - 6) * (self.process_stride - 6) + self.process_stride

                image_padded = tf.pad(image, [[0, pad_height - h], [0, pad_width - w]], mode='REFLECT')

                # Splitting and upsampling the windows
                # TensorFlow doesn't have an unfold function, but we can achieve something similar with tf.image.extract_patches
                sizes = [1, self.process_stride, self.process_stride, 1]
                strides = [1, self.process_stride - 6, self.process_stride - 6, 1]
                rates = [1, 1, 1, 1]  # For dilation, not used here
                windows = tf.image.extract_patches(images=tf.expand_dims(image_padded, 0),
                                                sizes=sizes,
                                                strides=strides,
                                                rates=rates,
                                                padding='VALID')
                windows = tf.reshape(windows, [-1, self.process_stride, self.process_stride, inputs.shape[-1]])
                windows = upsample(windows)
                outputs.extend(tf.reshape(windows, [-1, *windows.shape[2:]]))
                maxs.extend([tf.reduce_max(image)] * (windows.shape[0] * windows.shape[1]))
                mins.extend([tf.reduce_min(image)] * (windows.shape[0] * windows.shape[1]))

        # Note: Returning `windows.shape` directly after the loop doesn't make sense in this context,
        # because `windows` would only refer to the last processed batch. You might want to collect
        # these shapes in a list if needed for each image/window.
        return outputs, maxs, mins

    # Assuming `images_to_window_lists` and `model_tune` are adapted to TensorFlow
    def predict_sequence(self, inputs):
        counted_list = []
        eventsize_all = []
        inputs = tf.cast(inputs, tf.float32)
        counted_images = tf.zeros_like(inputs)

        image_cell_list, windowshape, maxs, mins = self.images_to_window_lists(inputs)
        for i, image_cell in enumerate(image_cell_list):

            if self.mode == 'dynamic_window':
                self.model_tune(image_cell)
            elif self.mode == 'dynamic_frame':
                # Similar logic adapted for TensorFlow
                pass  # Fill in based on adapted `model_tune`
            elif self.mode == 'static':
                # Ensure process_stride compatibility
                pass
            else:
                raise ValueError("Use mode = 'dynamic_window', 'dynamic_frame', or 'static'.")

            # Image cell thresholding and normalization
            image_cell = tf.where(image_cell < self.dark_threshold, 0, image_cell)
            image_cell_normalized = (image_cell - mins[i]) / (maxs[i] - mins[i])

            # Model inference (adapt based on your TensorFlow model)
            boxes = self.fastrcnn_model(image_cell_normalized[None, ...])[0]['boxes']

            select = []
            for row, value in enumerate(boxes):
                if 0 < (value[2] - value[0]) < 30 and 0 < (value[3] - value[1]) < 30:
                    select.append(row)
            select = torch.as_tensor(select, dtype=torch.int, device=self.device)
            filtered_boxes = torch.index_select(boxes, 0, select)
            filtered_boxes = filtered_boxes / 2.0
            image_cell_ori = F.interpolate(image_cell_ori[None, None, ...], scale_factor=0.5, mode='nearest')[0, 0]
            filtered, _, eventsize = self.locate(image_cell_ori, filtered_boxes)
            counted_list.append(filtered[None, ...])  # [1,w,h]
            eventsize_all = eventsize_all + eventsize

        image_num = int(len(counted_list) / windowshape[0] / windowshape[1])
        for index in range(image_num):
            counted_cells = counted_list[
                            index * windowshape[0] * windowshape[1]:(index + 1) * windowshape[0] * windowshape[1]]
            counted_cells = torch.cat(counted_cells)
            counted_cells = counted_cells.reshape(windowshape[0], windowshape[1], int(windowshape[2] / 2),
                                                  int(windowshape[3] / 2))
            counted_images[index] = stich_windows(counted_cells, k=3, cropx=inputs.shape[1], cropy=inputs.shape[2])

        return counted_images, eventsize_all

        
    

    def locate(self, image_array, boxes):
        width = 10
        filtered = tf.zeros_like(image_array)
        boxes = tf.cast(tf.round(boxes), tf.int32)
        coor = []
        eventsize = []

        for box in boxes:
            xarea = image_array[box[1]:(box[3] + 1), box[0]:(box[2] + 1)]

            # Padding logic adapted for TensorFlow
            if xarea.shape[0] > (width + 1) or xarea.shape[1] > (width + 1):
                paddings = tf.constant([[1, width], [1, width]])
                patch = tf.pad(xarea, paddings, mode='CONSTANT')[:width + 2, :width + 2]
            else:
                paddings = tf.constant([[1, width - xarea.shape[1] + 1], [1, width - xarea.shape[0] + 1]])
                patch = tf.pad(xarea, paddings, mode='CONSTANT')

            if self.method == 'max':
                model_x, model_y = tf.unravel_index(tf.argmax(patch, axis=None), patch.shape)
            elif self.method == 'binary_com':
                patch = tf.where(patch < 30, 0, 1)
                x = tf.range(0, patch.shape[0], dtype=tf.float32)
                y = tf.range(0, patch.shape[1], dtype=tf.float32)
                weights_x, weights_y = tf.meshgrid(x, y, indexing='ij')
                model_x = tf.math.round(tf.reduce_sum(patch * weights_x) / tf.reduce_sum(patch))
                model_y = tf.math.round(tf.reduce_sum(patch * weights_y) / tf.reduce_sum(patch))
            elif self.method == 'com':
                x = tf.range(0, patch.shape[0], dtype=tf.float32)
                y = tf.range(0, patch.shape[1], dtype=tf.float32)
                weights_x, weights_y = tf.meshgrid(x, y, indexing='ij')
                model_x = tf.math.round(tf.reduce_sum(patch * weights_x) / tf.reduce_sum(patch))
                model_y = tf.math.round(tf.reduce_sum(patch * weights_y) / tf.reduce_sum(patch))
            else:
                raise ValueError("Use 'max', 'com', or 'binary_com' to locate the entry position.")

            cx = model_x + box[1] - 1
            cy = model_y + box[0] - 1

            if cx >= image_array.shape[0] or cy >= image_array.shape[1] or cx < 0 or cy < 0:
                continue

            coor.append([cx, cy])
            eventsize.append(tf.reduce_sum(tf.cast(patch > 20, tf.int32)).numpy())

        coords = tf.cast(coor, tf.int32)
        for point in coords:
            filtered = tf.tensor_scatter_nd_add(filtered, [point], [1])

        return filtered, coords, eventsize
