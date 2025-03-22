import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Split the input into three channels
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)

        # Define a convolutional block
        def conv_block(input_tensor):
            x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
            return x

        # Apply the convolutional block to each channel group
        group1 = conv_block(split_tensor[0])
        group2 = conv_block(split_tensor[1])
        group3 = conv_block(split_tensor[2])

        # Combine the outputs from each group
        merged_features = Add()([group1, group2, group3])

        # Fuse the merged features with the original input
        combined_features = Add()([merged_features, input_layer])

        # Flatten and classify
        flatten_layer = Flatten()(combined_features)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model