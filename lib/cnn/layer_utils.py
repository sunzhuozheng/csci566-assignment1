from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        output_shape[0] = input_size[0]
        output_shape[1] = (input_size[1] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_shape[2] = (input_size[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_shape[3] = self.number_filters
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################

        #pad the input image according to self.padding (see np.pad)
        p = self.padding
        s = self.stride
        #N, out_h, out_w, C = output_shape
        img = np.pad(img, ((0, 0), (p, p), (p, p), (0, 0)), mode="constant", constant_values=0)
        self.meta = img

        #iterate over output dimensions, moving by self.stride to create the output
        output = np.zeros(output_shape)
        #for i in range(N):
            #for h in range(out_h):
                #for w in range(out_w):
                    #for c in range(C):
                        #vert_s = h * s
                        #vert_e = vert_s + self.kernel_size
                        #horiz_s = w * s
                        #horiz_e = horiz_s + self.kernel_size

                        #output[i, h, w, c] = np.sum(img[i, vert_s:vert_e, horiz_s:horiz_e, :] * self.params[self.w_name][:, :, :, c]) + self.params[self.b_name][c]
        for h in range(output_height):
            for w in range(output_width):
                vert_s = h * s
                vert_e = vert_s + self.kernel_size
                horiz_s = w * s
                horiz_e = horiz_s + self.kernel_size

                output[:, h, w, :] = np.sum(np.expand_dims(img[:, vert_s:vert_e, horiz_s:horiz_e, :], axis=4) * np.expand_dims(self.params[self.w_name], axis=0), axis=(1, 2, 3)) + self.params[self.b_name]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################

        _, H, W, _ = dprev.shape
        p = self.padding
        s = self.stride
        self.grads[self.w_name] = np.zeros(self.params[self.w_name].shape)
        dimg = np.zeros(img.shape)

        for h in range(H):
            for w in range(W):
                vert_s = h * s
                vert_e = vert_s + self.kernel_size
                horiz_s = w * s
                horiz_e = horiz_s + self.kernel_size

                img_slice = img[:, vert_s:vert_e, horiz_s:horiz_e, :]

                self.grads[self.w_name] += np.sum(np.expand_dims(img_slice, axis=4) * np.expand_dims(dprev[:, h, w, :], axis=(1, 2, 3)), axis=0)
                dimg[:, vert_s:vert_e, horiz_s:horiz_e, :] += np.sum(np.expand_dims(self.params[self.w_name], axis=0) * np.expand_dims(dprev[:, h, w, :], axis=(1,2,3)), axis=4)
        self.grads[self.b_name] = np.sum(dprev, axis=(0, 1, 2))
        dimg = dimg[:, p:-p, p:-p, :]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        N, input_H, input_W, C = img.shape
        s = self.stride
        output_H = (input_H - self.pool_size) // self.stride + 1
        output_W = (input_W - self.pool_size) // self.stride + 1
        output = np.zeros((N, output_H, output_W, C))

        #for i in range(N):
            #for h in range(output_H):
                #for w in range(output_W):
                    #for c in range(C):
                        #input_slice = img[i, h * self.stride:h * self.stride + self.pool_size, w * self.stride:w * self.stride + self.pool_size, c]
                        #output[i, h, w, c] = np.max(input_slice)
        for h in range(output_H):
            for w in range(output_W):
                vert_s = h * s
                vert_e = vert_s + self.pool_size
                horiz_s = w * s
                horiz_e = horiz_s + self.pool_size

                img_slice = img[:, vert_s:vert_e, horiz_s:horiz_e, :]
                output[:, h, w, :] = np.max(img_slice, axis=(1, 2))

        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        N, h_in, w_in, C = img.shape
        s = self.stride

        #for i in range(N):
            #for h in range(h_out):
                #for w in range(w_out):
                    #for c in range(C):
                        #in_slice = img[i, h * stride:h * stride + h_pool, w * stride:w * stride + w_pool, c]
                        #mask = (in_slice == np.max(in_slice))
                        #dimg[i, h * stride:h * stride + h_pool, w * stride:w * stride + w_pool, c] += mask * dprev[i, h, w, c]
        for h in range(h_out):
            for w in range(w_out):
                vert_s = h * s
                vert_e = vert_s + h_pool
                horiz_s = w * s
                horiz_e = horiz_s + w_pool

                img_slice = img[:, vert_s:vert_e, horiz_s:horiz_e, :]
                mask = (img_slice == np.expand_dims(np.max(img_slice, axis=(1, 2)), axis=(1, 2)))
                dimg[:, vert_s:vert_e, horiz_s:horiz_e, :] += mask * np.expand_dims(dprev[:, h, w, :], axis=(1, 2))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
