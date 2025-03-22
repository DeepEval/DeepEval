import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Add, Lambda, Multiply, GlobalAveragePooling2D
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3)) # CIFAR-10 dataset shape
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(input_layer)
    softmax = Lambda(lambda x: K.softmax(x))(conv1) # Get attention weights
    attention_weights = Multiply()([input_layer, softmax]) # Weighted processing

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(attention_weights)
    batch_norm = BatchNormalization()(conv2)
    act ='relu'
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(batch_norm)
    layer_norm = BatchNormalization()(conv3)
    act_layer = ReLU()(layer_norm)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(act_layer)
    concat_layer = Add()([conv4, attention_weights]) # Add the original input and processed output

    flatten_layer = Flatten()(concat_layer)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model