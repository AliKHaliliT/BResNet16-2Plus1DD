from tensorflow.keras.saving import register_keras_serializable # type: ignore
import tensorflow as tf
from math import ceil
from typing import Union, Any


@register_keras_serializable()
class _Conv2Plus1D(tf.keras.layers.Layer):

    """
    
    Conv2Plus1D layer from the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    Link: https://arxiv.org/abs/1711.11248v3

    The asset was sourced from the MobileViViT Repository.
    Link: https://github.com/AliKHaliliT/MobileViViT
    
    """

    def __init__(self, filters: int, kernel_size: tuple[int, int, int], 
                 strides: tuple[int, int, int], padding: str, 
                 use_bias: bool = True, **kwargs) -> None:

        """

        Constructor of the Conv2Plus1D layer.
        
        
        Parameters
        ----------
        filters : int
            Number of filters in the temporal convolutional layer.
            The number of filters in the spatial decomposition is calculated based on this value.
            See paper Sec. 3.5. for more details.

        kernel_size : tuple
            Kernel size of the convolutional layers.

        strides : tuple
            Strides of the convolutional layers.

        padding : str
            Padding of the convolutional layers.
                The options are:
                    `"valid"`
                        No padding.
                    `"same"`
                        Padding with zeros.

        use_bias : bool
            Bias term for the layer. The default is `True`.

        
        Returns
        -------
        None.
        
        """

        if not isinstance(filters, int) or filters < 0:
            raise ValueError(f"filters must be a non-negative integer. Received: {filters} with type {type(filters)}")
        if len(kernel_size) != 3 or not all(isinstance(k, int) and k > 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be a tuple of three positive integers. Received: {kernel_size} with type {type(kernel_size)}")
        if len(strides) != 3 or not all(isinstance(s, int) and s > 0 for s in strides):
            raise ValueError(f"strides must be a tuple of three positive integers. Received: {strides} with type {type(strides)}")
        if not isinstance(padding, str) or padding not in ["valid", "same"]:
            raise ValueError(f"padding must be either 'valid' or 'same'. Received: {padding} with type {type(padding)}")
        if not isinstance(use_bias, bool):
            raise TypeError(f"use_bias must be a boolean. Received: {use_bias} with type {type(use_bias)}")


        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias


    def build(self, input_shape: Union[tf.TensorShape, tuple[int, int, int, int, int]]) -> None:

        """

        Build method of the Conv2Plus1D layer.


        Parameters
        ----------  
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.


        Returns
        -------
        None.

        """

        super().build(input_shape)

        # Calcualting the number of filters required for the spatial decomposition
        spatial_filters = ceil((self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * input_shape[-1] * self.filters) /
                               ((self.kernel_size[1] * self.kernel_size[2] * input_shape[-1]) + (self.kernel_size[0] * self.filters)))
        
        # Spatial decomposition
        self.spatial_decompose = tf.keras.layers.Conv3D(filters=spatial_filters, 
                                                        kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                                                        strides=(1, self.strides[1], self.strides[2]), 
                                                        padding=self.padding,
                                                        use_bias=self.use_bias)
        
        # Temporal decomposition
        self.temporal_decompose = tf.keras.layers.Conv3D(filters=self.filters, 
                                                         kernel_size=(self.kernel_size[0], 1, 1), 
                                                         strides=(self.strides[0], 1, 1), 
                                                         padding=self.padding,
                                                         use_bias=self.use_bias)
        
        # Building the layers
        self.spatial_decompose.build(input_shape)
        self.temporal_decompose.build(self.spatial_decompose.compute_output_shape(input_shape))
                                

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Conv2Plus1D layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """


        return self.temporal_decompose(self.spatial_decompose(X))
    

    def compute_output_shape(self, input_shape: Union[tf.TensorShape, tuple[int, int, int, int, int]]) -> Union[tf.TensorShape, tuple[int, int, int, int, int]]:

        """
        
        Method to compute the output shape of the Conv2Plus1D layer.


        Parameters
        ----------
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor. 


        Returns
        -------
        output_shape : tuple
            Shape of the output tensor after applying the Conv2Plus1D layer. 
        
        """

        def compute_dim(dim, kernel, stride, padding):
            if padding == "same":
                return ceil(dim / stride)
            elif padding == "valid":
                return ceil((dim - kernel + 1) / stride)


        t = compute_dim(input_shape[1], self.kernel_size[0], self.strides[0], self.padding)
        h = compute_dim(input_shape[2], self.kernel_size[1], self.strides[1], self.padding)
        w = compute_dim(input_shape[3], self.kernel_size[2], self.strides[2], self.padding)


        return (input_shape[0], t, h, w, self.filters)


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Conv2Plus1D layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv2Plus1D layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias
        })


        return config
    

    def get_build_config(self) -> dict[str, Any]:

        """

        Method to get the build configuration of the Conv2Plus1D layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv2Plus1D layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias
        })
        

        return config