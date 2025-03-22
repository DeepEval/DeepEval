import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    def main_block(input_tensor):
        # Separable convolution with ReLU activation
        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # Batch normalization
        conv = BatchNormalization()(conv)
        return conv

    # Repeat main block three times
    main_output = input_layer
    for _ in range(3):
        main_output = main_block(main_output)
    
    # Branch pathway
    branch_output = Conv2D(filters=main_output.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Fusion of main and branch paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model