import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Split the input into three groups
        split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

        # Process each group with depthwise separable convolutions
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
        group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

        # Concatenate the outputs
        concat_layer = Concatenate(axis=-1)([group1, group2, group3])

        # Flatten and classify
        flatten_layer = Flatten()(concat_layer)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model