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

"""Contains KernelVariable, a variable which allows to manipulate a kernel on assign() and on read_value()."""

import tensorflow as tf
from tensorflow.python.distribute import ps_values as ps_distribute_values
from tensorflow.python.distribute import values as distribute_values


class KernelVariable(tf.Variable):
  """Variable that will cast itself to a different dtype in applicable contexts.
  """

  def __init__(self, variable, alter_kernel, upon_kernel_assign):
    """Creates an KernelVariable instance.

    Args:
      variable: A floating-point resource variable to wrap.

    Raises:
      ValueError: If `variable` is not a floating-point resource variable
    """
    if not isinstance(variable, tf.Variable):
      raise ValueError('variable must be of type tf.ResourceVariable, but got: '
                       '%s' % variable)
    if not variable.dtype.is_floating:
      raise ValueError('variable must be a floating point variable but has '
                       'type: %s' % variable.dtype.name)
    self._variable = variable
    # 'delegate' means AutoCastVariable.op return self._variable.op, which will
    # raise an AttributeError in Eager (as intended). If set to any other value,
    # AutoCastVariable.op returns that value instead, which is used to set the
    # op attribute in AutoCastVariable.assign().
    self._op = 'delegate'

    self.alter_kernel = alter_kernel
    self.upon_kernel_assign = upon_kernel_assign

  @property
  def dtype(self):
    """The dtype of the underlying variable, before any casts are done."""
    return self._variable.dtype

  def value(self):
    return self.alter_kernel(self._variable.value())

  def read_value(self):
    return self.alter_kernel(self._variable.read_value())

  def __getattr__(self, name):
    return getattr(self._variable, name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts this variable to a tensor."""
    return self.alter_kernel(tf.convert_to_tensor(self._variable, dtype, name, as_ref))

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def __repr__(self):
    if tf.executing_eagerly() and not self._in_graph_mode:
      repr_str = ("<KernelVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name}, '
                  'numpy={np_repr}>')
      return repr_str.format(v=self, np_repr=self.read_value())
    else:
      repr_str = ("<KernelVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name}>')
      return repr_str.format(v=self)

  # Method delegations: We delegate the following methods to self._variable.
  # Each of these methods simply calls the same method on self._variable. The
  # base Variable raises NotImplementedError for most of these, so we must
  # override them.
  #
  # We do not define the following methods from Variable for the following
  # reasons:
  #   * 'count_up_to': This method only applies to int variables, which cannot
  #     be wrapped with an AutoCastVariable.
  #   * 'ref': Instead we inherit the definition from Variable.
  #     If we defined and delegated to Variable, the ref of an AutoCastVariable
  #     would be the same as the ref of the underlying variable, which would be
  #     strange as they are different Python objects.

  def set_shape(self, shape):
    return self._variable.set_shape(self, shape)

  @property
  def trainable(self):
    return self._variable.trainable

  @property
  def synchronization(self):
    return self._variable.synchronization

  @property
  def aggregation(self):
    return self._variable.aggregation

  def eval(self, session=None):
    return self._variable.eval(session)

  def initialized_value(self):
    return self._variable.initialized_value()

  @property
  def initial_value(self):
    return self._variable.initial_value

  @property
  def constraint(self):
    return self._variable.constraint

  def _apply_assign_update(self,
                           update_fn,
                           value,
                           use_locking=None,
                           name=None,
                           read_value=True):
    # TODO(b/146181571): This logic can be simplified once
    # DistributedVariable.assign returns a DistributedVariable. Currently for
    # MirroredStrategy, it returns a Mirrored value.
    if tf.compat.v1.executing_eagerly_outside_functions():
      assign_op = update_fn(value, use_locking, name, False)
      if read_value:
        # We create a new AutoCastVariable with the same underlying tf.Variable.
        # The new AutoCastVariable is identical except the 'op' attribute is
        # defined. This matches the behavior of tf.Variable.assign.
        var = CreateKernelVariable(self.alter_kernel, self.upon_kernel_assign)(self._variable)
        var._op = assign_op  # pylint:disable=protected-access
        return var
      return assign_op

    # Fallback to wrapping the returned variable in graph mode if possible
    assign_var = update_fn(value, use_locking, name, read_value)
    if read_value:
      return CreateKernelVariable(self.alter_kernel, self.upon_kernel_assign)(self._variable)
    return assign_var

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # hook for calculation
    self.upon_kernel_assign(value)
    return self._apply_assign_update(self._variable.assign, value, use_locking,
                                     name, read_value)

  def load(self, value, session=None):
    return self._variable.load(value, session)

  @property
  def name(self):
    return self._variable.name

  @property
  def _shared_name(self):
    return self._variable._shared_name  # pylint:disable=protected-access

  @property
  def initializer(self):
    return self._variable.initializer

  @property
  def device(self):
    return self._variable.device

  @property
  def op(self):
    if self._op == 'delegate':
      return self._variable.op
    return self._op

  def _as_graph_element(self):
    graph_element = self._variable._as_graph_element()  # pylint:disable=protected-access
    if graph_element is None:
      return self._op
    return graph_element

  @property
  def graph(self):
    return self._variable.graph

  @property
  def shape(self):
    return self._variable.shape

  def get_shape(self):
    return self._variable.get_shape()

  def _gather_saveables_for_checkpoint(self):
    # By delegating this method to the wrapped variable, checkpoints with
    # AutoCastVariables are identical to checkpoints with normal variables.
    # Therefore models checkpointed with AutoCastVariables can be restored on
    # models with normal variables, and vice versa.
    return self._variable._gather_saveables_for_checkpoint()  # pylint:disable=protected-access

  def _map_resources(self, save_options):
    # By delegating this method to the wrapped variable, SavedModel with
    # AutoCastVariables are identical to SavedModel with normal variables.
    obj_map, resource_map = self._variable._map_resources(save_options)  # pylint:disable=protected-access
    obj_map[self] = obj_map[self._variable]
    return obj_map, resource_map

  # TODO(reedwm): Maybe encode the fact the variable is an KernelVariable in
  # to_proto().
  def to_proto(self, export_scope=None):
    return self._variable.to_proto(export_scope)

  def from_proto(self, variable_def, import_scope=None):
    return self._variable.from_proto(variable_def, import_scope)

  # Delegate the private attributes _handle_name and _initializer_op to
  # self._variable. SavedModel sets these attributes when loading a model. For
  # example, it sets _handle_name here:
  # https://github.com/tensorflow/tensorflow/blob/db26bd574fa95b5bdd53c08463dd19407cc0297e/tensorflow/python/keras/saving/saved_model/load.py#L211
  # We need to expose these attributes on AutoCastVariable as well for
  # SavedModel to work properly.
  # TODO(reedwm/kathywu): Find a better way to support SavedModel. Exposing
  # private attributes is hacky and difficult to maintain.
  @property
  def _handle_name(self):
    return self._variable._handle_name  # pylint: disable=protected-access

  @_handle_name.setter
  def _handle_name(self, handle_name):
    self._variable._handle_name = handle_name  # pylint: disable=protected-access

  @property
  def _initializer_op(self):
    return self._variable._initializer_op  # pylint: disable=protected-access

  @_initializer_op.setter
  def _initializer_op(self, initializer_op):
    self._variable._initializer_op = initializer_op  # pylint: disable=protected-access

  # Operator overloads:
  # Note we only overload operators that support floating-point types, as
  # non-float variables cannot be wrapped with an AutoCastVariable.
  # Also note: We call read_value() instead of value(), because value() causes
  # gradients not to work properly when TPUStrategy is used: b/143380936

  def __add__(self, o):
    return self.read_value() + o

  def __radd__(self, o):
    return o + self.read_value()

  def __sub__(self, o):
    return self.read_value() - o

  def __rsub__(self, o):
    return o - self.read_value()

  def __mul__(self, o):
    return self.read_value() * o

  def __rmul__(self, o):
    return o * self.read_value()

  def __truediv__(self, o):
    return self.read_value() / o

  def __rtruediv__(self, o):
    return o / self.read_value()

  def __floordiv__(self, o):
    return self.read_value() // o

  def __rfloordiv__(self, o):
    return o // self.read_value()

  def __mod__(self, o):
    return self.read_value() % o

  def __rmod__(self, o):
    return o % self.read_value()

  def __lt__(self, o):
    return self.read_value() < o

  def __le__(self, o):
    return self.read_value() <= o

  def __gt__(self, o):
    return self.read_value() > o

  def __ge__(self, o):
    return self.read_value() >= o

  def __getitem__(self, o):
    return self.read_value()[o]

  def __pow__(self, o, modulo=None):
    return pow(self.read_value(), o, modulo)

  def __rpow__(self, o):
    return pow(o, self.read_value())

  def __neg__(self):
    return -self.read_value()

  def __abs__(self):
    return abs(self.read_value())

  def __div__(self, o):
    try:
      return self.read_value().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self.read_value().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self.read_value().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self.read_value().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented


class CreateKernelVariable(object):
  """Creates an KernelVariable that wraps another variable.

  This typically just returns `AutoCastVariable(variable)`. But, if the variable
  is a DistributedVariable or one of its subclasses, we instead dynamically
  create a class that subclasses from both AutoCastVariable and
  variable.__class__. This is so the returned variable will still pass
  `isinstance(variable, variable.__class__)`, which is required for
  DistributedVariables and its subclasses to work properly.

  Args:
    variable: A floating-point resource variable to wrap.

  Returns:
    An AutoCastVariable that wraps the variable.
  """
  def __init__(self, alter_kernel, upon_kernel_assign):
    self.alter_kernel = alter_kernel
    self.upon_kernel_assign = upon_kernel_assign

  def __call__(self, variable):
    if not isinstance(variable, (distribute_values.DistributedVariable,
                                 ps_distribute_values.AggregatingVariable)):
      return KernelVariable(variable, self.alter_kernel, self.upon_kernel_assign)

    class KernelDistributedVariable(KernelVariable, variable.__class__):
      """An KernelVariable that also subclasses from variable.__class__.

      variable.__class__ is either a DistributedVariable or an
      AggregatingVariable.
      """

      def __repr__(self):
        if issubclass(ps_distribute_values.AggregatingVariable,
                      variable.__class__):
          # AggregatingVariable's __repr__ simply calls super.__repr__. So we do
          # the same here for consistency, which calls AutoCastVariable.__repr__.
          return super(KernelDistributedVariable, self).__repr__()

        # pylint: disable=missing-format-attribute
        return ('<KernelDistributedVariable dtype={v.dtype.name} '
                'inner_variable={v._variable}>'
               ).format(v=self)
        # pylint: enable=missing-format-attribute

    return KernelDistributedVariable(variable, self.alter_kernel, self.upon_kernel_assign)
