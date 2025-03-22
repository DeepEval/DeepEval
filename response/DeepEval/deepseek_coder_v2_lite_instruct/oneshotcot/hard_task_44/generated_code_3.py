import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input into three groups
        split1 = Lambda(lambda x: x[:, :, :, :10])(input_tensor)
        split2 = Lambda(lambda x: x[:, :, :, 10:20])(input_tensor)
        split3 = Lambda(lambda x: x[:, :, :, 20:])(input_tensor)
        
        # Feature extraction for each group
        conv1 = Conv2D(32, (1, 1), activation='relu')(split1)
        conv2 = Conv2D(32, (3, 3), activation='relu')(split2)
        conv3 = Conv2D(32, (5, 5), activation='relu')(split3)
        
        # Dropout to reduce overfitting
        dropout = Dropout(0.25)(conv3)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1, conv2, dropout])
        return concatenated

    block1_output = block1(input_layer)
    batch_norm = BatchNormalization()(block1_output)

    # Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        
        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch2a = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        branch2b = Conv2D(32, (3, 3), activation='relu')(branch2a)
        
        # Branch 3: <1x1 convolution, 5x5 convolution>
        branch3a = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        branch3b = Conv2D(32, (5, 5), activation='relu')(branch3a)
        
        # Branch 4: <3x3 max pooling, 1x1 convolution>
        branch4a = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
        branch4b = Conv2D(32, (1, 1), activation='relu')(branch4a)
        
        # Concatenate the outputs from all branches
        concatenated = Concatenate()([branch1, branch2b, branch3b, branch4b])
        return concatenated

    block2_output = block2(batch_norm)
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()