import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Input shape for CIFAR-10

    # First block for feature extraction
    def feature_extraction(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(conv2)
        norm1 = BatchNormalization()(conv3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)
        split1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=4), output_shape=(64, 16, 16, 3))(pool1)

        return split1

    split1 = feature_extraction(input_tensor=input_layer)
    split2 = feature_extraction(input_tensor=input_layer)
    split3 = feature_extraction(input_tensor=input_layer)

    # Concatenate outputs from each split
    concat = Concatenate(axis=4)([split1, split2, split3])

    # Second block for additional feature extraction
    def additional_features(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(4, 4), strides=(1, 1))(conv2)
        split2 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=2, axis=4), output_shape=(64, 8, 8, 3))(avg_pool)

        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
        max_pool = MaxPooling2D(pool_size=(4, 4), strides=2, padding='same')(conv5)
        split3 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=4), output_shape=(64, 4, 4, 3))(max_pool)

        return Concatenate(axis=4)([split2, split3])

    concat = additional_features(concat)

    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(concat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()