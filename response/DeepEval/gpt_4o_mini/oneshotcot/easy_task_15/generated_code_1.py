import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def feature_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(conv1x1_2)
        dropout = Dropout(rate=0.5)(avg_pool)
        
        return dropout

    # First feature block
    block1_output = feature_block(input_layer)
    # Second feature block
    block2_output = feature_block(block1_output)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    # Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model