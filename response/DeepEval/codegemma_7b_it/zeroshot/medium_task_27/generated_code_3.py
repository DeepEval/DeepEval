from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutional layer
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)

    # Branch 2: 5x5 convolutional layer
    branch_2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(inputs)

    # Global average pooling layer
    gap = GlobalAveragePooling2D()(Add()([branch_1, branch_2]))

    # Attention layers
    attention_layer_1 = Dense(128, activation='relu')(gap)
    attention_layer_2 = Dense(10, activation='softmax')(attention_layer_1)

    # Weighted output
    weighted_output = Multiply()([branch_1, attention_layer_2[:, 0:1]])
    weighted_output = Add()([weighted_output, Multiply()([branch_2, attention_layer_2[:, 1:2]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_1, attention_layer_2[:, 2:3]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_2, attention_layer_2[:, 3:4]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_1, attention_layer_2[:, 4:5]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_2, attention_layer_2[:, 5:6]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_1, attention_layer_2[:, 6:7]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_2, attention_layer_2[:, 7:8]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_1, attention_layer_2[:, 8:9]])])
    weighted_output = Add()([weighted_output, Multiply()([branch_2, attention_layer_2[:, 9:10]])])

    # Output layer
    outputs = Dense(10, activation='softmax')(weighted_output)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model