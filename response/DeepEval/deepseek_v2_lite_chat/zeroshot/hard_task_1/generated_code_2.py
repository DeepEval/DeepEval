import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate, Dense, Activation, Add, Concatenate
from keras.layers import Layer, Conv2DTranspose, Reshape, Conv2D, MaxPooling2D, Conv2DLayer, Permute
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Define the input shape
    input_shape = (32, 32, 3)  # Input image shape

    # Create the base model from VGG16, excluding the last three layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Modify the base model to not include the fully connected layers
    for layer in base_model.layers:
        layer.trainable = False

    # Define the model inputs and outputs
    input_tensor = Input(shape=input_shape)
    base_model_output = base_model(input_tensor)

    # Path 1: Global average pooling, two fully connected layers
    avg_pool = GlobalAveragePooling2D()(base_model_output)
    fc1 = Dense(1024, activation='relu')(avg_pool)
    fc2 = Dense(1024, activation='relu')(fc1)

    # Path 2: Global max pooling, two fully connected layers
    max_pool = GlobalMaxPooling2D()(base_model_output)
    fc1 = Dense(1024, activation='relu')(max_pool)
    fc2 = Dense(1024, activation='relu')(fc1)

    # Concatenate the outputs of Path 1 and Path 2
    concat = concatenate([fc1, fc2])

    # Channel attention module
    channel_attention = Dense(256, activation='relu')(concat)
    channel_attention = Dense(128, activation='relu')(channel_attention)
    sigmoid_channel_attention = Activation('sigmoid')(channel_attention)

    # Multiply the base model output with the channel attention weights
    att_base_output = Activation('sigmoid')(sigmoid_channel_attention) * base_model_output

    # Spatial attention module
    spatial_attention = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(base_model_output)

    # Concatenate the channel and spatial attention features
    concat_att = concatenate([att_base_output, spatial_attention])

    # Extraction of spatial features
    avg_pool_spatial = MaxPooling2D(pool_size=(3, 3))(input_tensor)
    max_pool_spatial = MaxPooling2D(pool_size=(3, 3))(input_tensor)
    concat_spatial = concatenate([avg_pool_spatial, max_pool_spatial])

    # 1x1 convolution and sigmoid activation
    norm_spatial = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), activation='sigmoid', padding='same')(concat_spatial)

    # Multiply the channel attention features with the spatial attention features
    final_output = Activation('sigmoid')(concat_att) * norm_spatial

    # Classify using a fully connected layer
    output = Dense(10, activation='softmax')(final_output)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    # Print model summary
    model.summary()

    return model

# Call the function
model = dl_model()