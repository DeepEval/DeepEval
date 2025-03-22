from tensorflow.keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define input shape based on CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Main Path
    # 7x7 Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(inputs)
    # Layer Normalization
    x = LayerNormalization()(x)
    # Two sequential 1x1 Pointwise Convolutions
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)
    
    # Branch Path (direct connection to input)
    branch = inputs
    
    # Combine outputs of both paths through addition
    combined = Add()([x, branch])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Two fully connected layers for classification
    fc1 = Dense(units=256, activation='relu')(flattened)
    outputs = Dense(units=10, activation='softmax')(fc1)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Usage example (compiling and checking model summary)
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()