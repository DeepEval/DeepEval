import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def feature_extractor(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
        max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

        return max_pooling2

    def refine_features(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling)
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
        max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

        return max_pooling2

    extracted_features = feature_extractor(input_tensor=input_layer)
    refined_features = refine_features(input_tensor=extracted_features)

    flatten = Flatten()(refined_features)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model