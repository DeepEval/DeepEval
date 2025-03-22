from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Concatenate, Reshape, Add
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the branch block
    def branch_block(x):
        # Elevate the dimension through a 1x1 convolutional layer
        x_elevated = Conv2D(6, (1, 1), activation='relu', name='branch_block_elevated')(x)
        
        # Extract features through a 3x3 depthwise separable convolutional
        x_features = DepthwiseConv2D((3, 3), activation='relu', name='branch_block_features')(x_elevated)
        
        # Reduce the dimension through a 1x1 convolutional layer
        x_reduced = Conv2D(6, (1, 1), activation='relu', name='branch_block_reduced')(x_features)
        
        # Add the block's input to form the output
        output = Add()([x, x_reduced])
        return output

    # Define the three branches
    branch1_output = branch_block(input_layer)
    branch2_output = branch_block(branch1_output)
    branch3_output = branch_block(branch2_output)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate(axis=-1)([branch1_output, branch2_output, branch3_output])

    # Reshape the concatenated output to 1D
    reshaped = Reshape((3 * 6 * 28 * 28,))(concatenated)

    # Pass the reshaped output through a fully connected layer to generate classification probabilities
    output_layer = Dense(10, activation='softmax')(reshaped)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()