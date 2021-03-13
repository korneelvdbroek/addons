# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.layers.kernel_variable import kernel_variable
from tensorflow_addons.layers import spectral_normalization
from tensorflow_addons.utils import test_utils
from tensorflow.python.keras.engine import base_layer_utils


class DiagKernel(tf.keras.layers.Layer):
  def __init__(self,
               kernel_initializer,
               trainable=True,
               name=None,
               getter=None,
               **kwargs):
    super(DiagKernel, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs)
    self.kernel_initializer = kernel_initializer
    self.getter = getter

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[input_shape[-1], input_shape[-1]],
                                  initializer=self.kernel_initializer,
                                  getter=self.getter)
    print(self.kernel)

  def call(self, input, **kwargs):
    print(self.kernel)
    return tf.matmul(input, tf.linalg.diag(tf.linalg.diag_part(self.kernel)))


class TensorInitializer(tf.keras.initializers.Initializer):
  def __init__(self, tensor):
    super().__init__()
    self.tensor = tensor

  def __call__(self, shape, dtype=None, **kwargs):
    return self.tensor


class SpectralNormalizer(object):
  def __init__(self,
               input_channels,
               output_channels,
               power_iterations: int = 1,
               dtype=None):
    super().__init__()
    if power_iterations <= 0:
      raise ValueError(
        "`power_iterations` should be greater than zero, got "
        "`power_iterations={}`".format(power_iterations)
      )
    self.power_iterations = power_iterations

    self._set_dtype_policy(dtype)

    self.output_channels = output_channels
    self.input_channels = input_channels

    self._u = tf.Variable(tf.random.truncated_normal(shape=(1, self.output_channels), stddev=0.02),
                          trainable=False,
                          dtype=self.dtype,
                          name="sn_u")

    self._v = tf.Variable(tf.zeros(shape=(1, input_channels)),
                          trainable=False,
                          dtype=self.dtype,
                          name="sn_v")

  def _set_dtype_policy(self, dtype):
    """Sets self._dtype_policy.
    Code borrowed from python.keras.engine.base_layer.py
    """
    if isinstance(dtype, tf.keras.mixed_precision.Policy):
      self._dtype_policy = dtype
    elif isinstance(dtype, dict):
      self._dtype_policy = tf.keras.utils.deserialize_keras_object(dtype)
    elif dtype:
      self._dtype_policy = tf.keras.mixed_precision.Policy(tf.dtypes.as_dtype(dtype).name)
    else:
      self._dtype_policy = tf.keras.mixed_precision.global_policy()

  @property
  def dtype(self):
    """The dtype of the constraint weights.
    """
    return self._dtype_policy.variable_dtype

  @property
  def compute_dtype(self):
    """The dtype of the constraints' computations.
    """
    return self._dtype_policy.compute_dtype

  @tf.function
  def _compute_uv(self, w):
    """Computes and stores auxiliary variables u and v, using kernel as input
    """
    # down-cast
    w_flat = tf.reshape(tf.cast(w, dtype=self.compute_dtype), [-1, self.output_channels])
    u = tf.cast(self._u, dtype=self.compute_dtype)

    with tf.name_scope("spectral_normalize"):
      for _ in range(self.power_iterations):
        v = tf.math.l2_normalize(tf.matmul(u, w_flat, transpose_b=True))
        u = tf.math.l2_normalize(tf.matmul(v, w_flat))

      # up-cast & update variables
      self._u.assign(tf.cast(u, dtype=self.dtype))
      self._v.assign(tf.cast(v, dtype=self.dtype))

  def spectral_normalize(self, w, epsilon=1e-12):
    w_flat = tf.reshape(w, [-1, self.output_channels])
    u = tf.cast(self._u, dtype=self.compute_dtype)
    v = tf.cast(self._v, dtype=self.compute_dtype)

    sigma = tf.matmul(tf.matmul(v, w_flat), u, transpose_b=True)
    return tf.sign(sigma) * w / tf.maximum(tf.abs(sigma), epsilon)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_differentiation():
  spectralnormalizer = SpectralNormalizer(2, 2)

  def alter_kernel(weights):
    return spectralnormalizer.spectral_normalize(weights)

  def upon_kernel_assign(weights):
    spectralnormalizer._compute_uv(weights)

  def get_kernel_variable(*args, **kwargs):
    variable = base_layer_utils.make_variable(*args, **kwargs)
    return kernel_variable.CreateKernelVariable(alter_kernel, upon_kernel_assign)(variable)
  getter = get_kernel_variable

  inputs = tf.keras.layers.Input(shape=[2])

  w00 = 2.
  w11 = 1.
  w = tf.constant([[w00, 0.], [0., w11]])
  sn_layer = DiagKernel(kernel_initializer=TensorInitializer(w), getter=getter)

  model = tf.keras.models.Sequential(layers=[inputs, sn_layer])

  x0 = x1 = 1
  x = tf.constant([[x0, x1]], dtype=tf.float32)

  # help spectral normalization in finding largest eigenvalue by setting u
  _ = model(x)
  sn_layer.u.assign(tf.constant([[1., 0.]]))

  with tf.GradientTape() as tape:
    y = model(x, training=True)
    loss = tf.reduce_sum(y)

  # loss_target = (w00 x0 + w11 x1) / w00
  loss_target = tf.reduce_sum(tf.matmul(w, x, transpose_b=True)) / tf.reduce_max(w)
  assert loss == loss_target

  gradients = tape.gradient(loss, model.trainable_variables)
  gradients_target = [[-w11/w00**2 * x1, 0.], [0., x1 / w00]]

  assert np.testing.assert_allclose(gradients, gradients_target)
