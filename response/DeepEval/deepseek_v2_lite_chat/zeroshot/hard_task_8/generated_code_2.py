import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, concatenate, Reshape, Permute, Flatten
from keras.layers import Layer

def custom_depthwise_conv2d(inputs, kernel_size):
    depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, depth_multiplier=1,
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    bn = keras.layers.BatchNormalization()(depthwise_conv)
    return keras.layers.ReLU()(bn)

class DepthwiseSeparableConv2D(Layer):
    def __init__(self, **kwargs):
        super(DepthwiseSeparableConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        depthwise_kernel = self.add_weight(name='depthwise_kernel', shape=(3, 3, 1, 32),
                                           initializer='uniform', trainable=True)
        pointwise_kernel = self.add_weight(name='pointwise_kernel', shape=(1, 1, 32, 16),
                                           initializer='uniform', trainable=True)
        self.built = True

    def call(self, x):
        if not self.built:
            raise ValueError(
                'Layer was never built. Call `build` before calling `call` first.')
        return custom_depthwise_conv2d(x, 3)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1], input_shape[0][-1], input_shape[1][-1], input_shape[3][-1]

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1
    x = custom_depthwise_conv2d(inputs, 1)(inputs)
    x = DepthwiseSeparableConv2D()(x)
    x = custom_depthwise_conv2d(x, 3)(x)
    x = DepthwiseSeparableConv2D()(x)
    
    # Branch path
    branch_x = custom_depthwise_conv2d(inputs, 1)(inputs)
    branch_x = DepthwiseSeparableConv2D()(branch_x)
    
    # Concatenate both paths
    merged = concatenate([x, branch_x])
    
    # Block 2
    shape = K.int_shape(merged)
    new_shape = (shape[1], shape[2], shape[3]*2, shape[3])
    reshaped = Reshape(new_shape)(merged)
    reshaped = Permute((2, 3, 1))(reshaped)
    reshaped = Reshape(new_shape)(reshaped)
    
    # Fully connected layer
    fc = Dense(10, activation='softmax')(reshaped)
    
    # Model
    model = Model(inputs=inputs, outputs=fc)
    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])