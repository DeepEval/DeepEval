import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input

def dl_model():     
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Lambda layer for multi-scale feature extraction
    def multi_scale_feature_extraction(input_tensor):
        # Split the input tensor into three groups along the channel dimension
        channels = input_tensor.shape[-1]
        groups = tf.split(input_tensor, channels, axis=-1)
        
        # Apply different convolutional kernels: 1x1, 3x3, and 5x5
        conv1x1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(groups[0])
        conv3x3 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(groups[1])
        conv5x5 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(groups[2])
        
        # Concatenate the outputs from these three groups
        output_tensor = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
        
        return output_tensor
    
    # Apply multi-scale feature extraction
    multi_scale_output = multi_scale_feature_extraction(input_layer)
    
    # Batch normalization and pooling
    batch_norm = layers.BatchNormalization()(multi_scale_output)
    max_pooling = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm)
    
    # Flatten the output
    flatten_layer = layers.Flatten()(max_pooling)
    
    # Two fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flatten_layer)
    dense2 = layers.Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))