# coding=utf-8


import math
import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense
registry = {}
register = partial(register, registry=registry)


@register('none')
def null_connection(x, _, __):
    return x


@register('residual')
def residual(x, res, _):
    if x.shape[-1] != res.shape[-1]:
        x = dense(x, res.shape.as_list()[-1], name='residual_projection')
    return (x + res) * math.sqrt(0.5)


@register('aug')
def augmented_residual(x, res, i):
    if i == 1:
        x = tf.concat([res, x], axis=-1)  # res is embedding
    elif i > 1:
        hidden_size = int(x.shape[-1])
        x = (res[:, :, -hidden_size:] + x) * math.sqrt(0.5)
        x = tf.concat([res[:, :, :-hidden_size], x], axis=-1)  # former half of res is embedding
    return x
