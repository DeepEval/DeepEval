import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Step 1: Split the input into 3 groups along the channel dimension
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Step 2: Apply 1x1 convolutions to each group
    conv_groups = []
    for group in split_groups:
        conv = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv_groups.append(conv)
    
    # Step 3: Downsample each group using average pooling
    pooled_groups = [AveragePooling2D(pool_size=(2, 2), strides=2)(conv) for conv in conv_groups]
    
    # Step 4: Concatenate the pooled feature maps along the channel dimension
    concatenated = Concatenate(axis=-1)(pooled_groups)

    # Step 5: Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Step 6: Add two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model