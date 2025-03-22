import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    dropout = Dropout(0.5)(pool)

    # Branch Pathway
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(input_layer)

    # Fusion
    adding_layer = Add()([dropout, conv4])

    # Global Average Pooling, Flatten and Dense
    global_avg_pool = GlobalAveragePooling2D()(adding_layer)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model