import keras
from keras.layers import Input, Conv2D, Lambda, LayerNormalization, ReLU, Add, Multiply, Softmax
from keras.models import Model

def attention_block(input_tensor, epsilon=1e-6):
    kernel_size = 1
    filters = input_tensor.shape[-1]
    # 1x1 convolution for generating attention weights
    attention = Conv2D(filters, (kernel_size, kernel_size), padding="same")(input_tensor)
    attention = Softmax()(attention + epsilon)  # Softmax normalization
    attention_weighted_input = Multiply()([input_tensor, attention])
    return attention_weighted_input

def reduce_dim_block(input_tensor, reduction_factor=3):
    filters = input_tensor.shape[3]
    # Reduce dimensionality to one-third
    reduced = Conv2D(filters // reduction_factor, (1, 1))(input_tensor)
    reduced = LayerNormalization(epsilon=1e-6)(reduced)
    reduced = ReLU()(reduced)
    # Expand dimensionality back to original size
    expanded = Conv2D(filters, (1, 1), padding="same")(reduced)
    expanded = LayerNormalization(epsilon=1e-6)(expanded)
    expanded = ReLU()(expanded)
    return expanded

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Attention block
    attention_out = attention_block(inputs)
    
    # Reduce and expand dimensions
    reduced = reduce_dim_block(attention_out)
    expanded = reduce_dim_block(reduced, reduction_factor=3)  # Reduce then expand back
    
    # Add original input for contextual information
    attention_and_original = Add()([expanded, inputs])
    
    # Flatten and fully connected layers
    flattened = Flatten()(attention_and_original)
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

model = dl_model()
model.summary()