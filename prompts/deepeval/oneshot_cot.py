class OneshotCoTPromptDesigner:
    """
    OneshotCoTPromptDesigner: A class to design a prompt for a task using the chain-of-thinking paradigm.
    Prompt:prefix-<one example(requirement,chain-of-thinking,code),new requirement> --> code.
    """
    def __init__(self):
        pass

    def prompt(self, task_requirement):

        background = """
    As a developer specializing in deep learning, you are expected to complete the code 
using Functional APIs of Keras, ensuring it meets the requirement of a task. You could draw 
inspiration from the provided example.
    """

        example_task = """
    The requirement of an example task is as follows: 
    "Please create a deep learning model for image classification using the MNIST dataset. The model should include
a convolutional layer followed by a pooling layer, both connected in series. Subsequently, implement a specific
block featuring four parallel paths: a 1x1 convolution, a 3x3 convolution, a 5x5 convolution, and a 1x1 max pooling
layer. Concatenate the outputs of these paths. Then, apply batch normalization and flatten the result. Finally,
the output should pass through three fully connected layers to produce the final classification."
    """
        cot = """
    The process for completing the example task is as follows:
    1.first of all: import keras and all necessary packages;
    2.secondly: construct the model with the following steps:
        step 1: add input layer;
        step 2: add convolutional layer;
        step 3: add maxpooling layer;
        step 4: define a block;
            step 4.1: add convolutional layer as the first path;
            step 4.2: add convolutional layer as the second path;
            step 4.3: add convolutional layer as the third path;
            step 4.4: add maxpooling layer as the fourth path;
            step 4.5: add concatenate layer to merge the above paths;
        step 5: add batch normalization layer;
        step 6: add flatten layer;
        step 7: add dense layer;
        step 8: add dense layer;
        step 9: add dense layer;
    3.lastly: build the model
    """
        example_code = """
    The completed code for the example task is as follows:
    ```python
    import keras
    from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

    def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

        def block(input_tensor):

            path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
            output_tensor = Concatenate()([path1, path2, path3, path4])

            return output_tensor
        
        block_output = block(input_tensor=max_pooling)
        bath_norm = BatchNormalization()(block_output)
        flatten_layer = Flatten()(bath_norm)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model
    ```
    """
  
        new_task_requirement = (
            """
    Please refer to above example and complete new task detailed as follows:\n"""
            + "    "
            + f'"{task_requirement}"'
        )

        code_format = """ 
    Please import Keras and all necessary packages, then complete python code in the 'dl_model()' function and return the constructed 'model'.
    ```python
    def dl_model():
        
        return model
    ```
    """
        prompt = (
            background
            + example_task
            + cot
            + example_code        
            + new_task_requirement
            + code_format
        )

        return prompt