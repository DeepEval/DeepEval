import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split the input into three groups
        split_1 = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=3))(x)
        # Process each group with a different kernel size
        conv_groups = [
            Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_1[0]),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_1[1]),
            Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_1[2])
        ]
        # Concatenate the outputs of the three groups
        x = Concatenate(axis=3)(conv_groups)
        # Batch normalization
        x = BatchNormalization()(x)
        return x

    x = block1(input_layer)

    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x))
        
        # Path 3: 1x1 convolution followed by two sub-paths
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        sub_path3_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path3)
        sub_path3_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path3)
        path3 = Concatenate()([sub_path3_1, sub_path3_2])
        
        # Path 4: 1x1 convolution followed by 3x3 convolution, then two sub-paths
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        sub_path4_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path4)
        sub_path4_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path4)
        path4 = Concatenate()([sub_path4_1, sub_path4_2])
        
        # Concatenate the outputs of the four paths
        x = Concatenate()([path1, path2, path3, path4])
        return x

    x = block2(x)

    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()