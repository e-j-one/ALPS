import jax.numpy as jnp
import flax.nnx as nnx
from typing import Tuple


class MLP(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, rngs: nnx.Rngs):
        # MLP layers for dynamics prediction
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nnx.Linear(current_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.relu)
            current_dim = hidden_dim

        layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)
    
    def __call__(self, input):
        return self.net(input)
    

class CNNEncoder(nnx.Module):
    """CNN encoder for image observations"""
    def __init__(self, input_shape: Tuple[int, ...], hidden_dim: int, rngs: nnx.Rngs):
        self.input_shape = input_shape

        # conv layers
        self.conv1 = nnx.Conv(input_shape[-1], 16, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)

        # compute output shapes via dummy forward pass
        dummy_input = jnp.zeros((1, *input_shape))
        out1 = self.conv1(dummy_input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        # store intermediate shapes for decoder (H, W, C)
        self.intermediate_shapes = [
            out1.shape[1:],  # after conv1
            out2.shape[1:],  # after conv2
        ]
        self.conv_output_shape = out3.shape[1:]
        conv_output_size = int(jnp.prod(jnp.array(self.conv_output_shape)))

        self.flatten_linear = nnx.Linear(conv_output_size, hidden_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.flatten_linear(x)
        return x


class CNNDecoder(nnx.Module):
    """CNN decoder for reconstructing image observations"""
    def __init__(self, output_shape: Tuple[int, ...], hidden_dim: int, conv_output_shape: Tuple[int, ...], intermediate_shapes: list, rngs: nnx.Rngs):
        self.output_shape = output_shape
        self.conv_output_shape = conv_output_shape
        self.intermediate_shapes = intermediate_shapes  # shapes from encoder for cropping
        conv_output_size = int(jnp.prod(jnp.array(conv_output_shape)))

        self.unflatten_linear = nnx.Linear(hidden_dim, conv_output_size, rngs=rngs)

        # deconv layers mirror the encoder
        self.deconv1 = nnx.ConvTranspose(32, 32, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(32, 16, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(16, output_shape[-1], kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)

    def __call__(self, x):
        batch_size = x.shape[0]
        x = nnx.relu(self.unflatten_linear(x))
        x = jnp.reshape(x, (batch_size, *self.conv_output_shape))

        # deconv1
        x = nnx.relu(self.deconv1(x))
        target_h, target_w = self.intermediate_shapes[1][:2]
        x = x[:, :target_h, :target_w, :]

        # deconv2
        x = nnx.relu(self.deconv2(x))
        target_h, target_w = self.intermediate_shapes[0][:2]
        x = x[:, :target_h, :target_w, :]

        # deconv3: crop to match original input shape
        x = self.deconv3(x)
        target_h, target_w = self.output_shape[:2]
        x = x[:, :target_h, :target_w, :]

        return nnx.sigmoid(x)   # because input images are normalized between 0 and 1
