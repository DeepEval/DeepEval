import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_conv2)
    main_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv3)
    main_dropout = Dropout(rate=0.5)(main_pooling)

    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing the outputs from both pathways
    fused_output = Add()([main_dropout, branch_conv])

    # Final layers
    global_avg_pooling = GlobalAveragePooling2D()(fused_output)
    flatten_layer = Flatten()(global_avg_pooling)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model