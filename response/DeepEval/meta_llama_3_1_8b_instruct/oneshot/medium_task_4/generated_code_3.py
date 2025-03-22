import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features two pathways that combine to create a comprehensive feature representation through addition:
    path1 consists of two blocks of convolution followed by average pooling, which progressively extracts deep features from the images.
    Path2 employs a single convolutional layer to process the input.
    After feature extraction, the outputs from both pathways are flattened into a one-dimensional vector.
    This vector is then mapped to a probability distribution over the 10 classes using a fully connected layer.
    
    :return: The constructed deep learning model
    """

    input_layer = Input(shape=(32, 32, 3))
    
    # Define path1: two blocks of convolution followed by average pooling
    def path1_block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
        return avg_pool
    
    block1 = path1_block(input_layer, 32)
    block2 = path1_block(block1, 64)
    
    # Define path2: a single convolutional layer
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both pathways through addition
    combined = Add()([block2, path2])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(bath_norm)
    
    # Map the flattened vector to a probability distribution over the 10 classes using a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model