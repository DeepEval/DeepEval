# Import necessary packages
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Average pooling layer 1 with window size 1x1 and stride 1x1
    x = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), name='avg_pool_1')(inputs)
    
    # Average pooling layer 2 with window size 2x2 and stride 2x2
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avg_pool_2')(x)
    
    # Average pooling layer 3 with window size 4x4 and stride 4x4
    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), name='avg_pool_3')(x)
    
    # Flatten the output of the pooling layers
    x = Flatten(name='flatten')(x)
    
    # Concatenate the flattened outputs
    x = concatenate([x, Flatten()(x)], axis=1, name='concatenate')
    
    # Fully connected layer 1 with 128 units
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='fc1')(x)
    
    # Flatten the output of the first fully connected layer
    x = Flatten()(x)
    
    # Fully connected layer 2 for classification with 10 units (CIFAR-10)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model