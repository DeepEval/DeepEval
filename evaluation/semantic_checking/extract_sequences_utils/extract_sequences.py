class ExtractSequences():
    """
    Extract API call sequences, data call sequences, API call sets from code
    """
    def __init__(self):
        pass

    def extract_code_flatten_info(self, code_info):
        """
        Flatten the code containing branch structure and extract the API call relationship in the code
        """
        try:
            code_flatten_infos = []
            for i in range(len(code_info)):
                source_nodes = code_info[i]['Info']['source_node']
                target_node = code_info[i]['Info']['target_node']
                if isinstance(source_nodes, list):
                    for source_node in source_nodes:
                        code_flatten_infos.append([(source_node, target_node), code_info[i]['Info']['DL API']])
                else:
                    code_flatten_infos.append([(source_nodes, target_node), code_info[i]['Info']['DL API']])
        except Exception as e:
            print("error in extract_code_flatten_info", e)
            code_flatten_infos = []
            raise

        return code_flatten_infos

    def extract_data_sequence(self, code_info, start_node, ending_node):
        """
        Extract data call sequences in the code
        """
        try:
            def generate_api_sequences(code_info_flatten, current_sequence, ending_node, data_flow):
                current_node = current_sequence[-1]  
                if current_node in ending_node:
                    data_flow.append(tuple(current_sequence.copy()))
                    return data_flow
                for i in range(len(code_info_flatten)):
                    source_node, target_node = code_info_flatten[i][0]
                    if source_node == current_node:
                        current_sequence.append(target_node)
                        generate_api_sequences(code_info_flatten, current_sequence, ending_node, data_flow)
                        current_sequence.pop()
                return data_flow

            code_info_flatten = self.extract_code_flatten_info(code_info)
            data_sequences = []
            if code_info_flatten != []:
                data_sequences = generate_api_sequences(code_info_flatten, start_node, ending_node, data_sequences)

        except Exception as e:
            print("error in extract_data_sequence", e)
            raise
        return data_sequences

    def extract_api_sequence(self, code_info, start_node, ending_node):
        """
        Extract API call sequences in the code
        """
        data_sequences = self.extract_data_sequence(code_info, start_node, ending_node)
        code_flatten_infos = self.extract_code_flatten_info(code_info)
        api_sequences = []
        try:
            if data_sequences != []:
                for data_sequence in data_sequences:
                    mapped_api_sequence = []
                    for i in range(len(data_sequence) - 1):
                        input_output_pairs = [(data_sequence[i], data_sequence[i + 1])]
                        for pair in input_output_pairs:
                            for code_flatten_info in code_flatten_infos:
                                if pair in code_flatten_info:
                                    for key in code_flatten_info[1]:
                                        if key != None:
                                            mapped_api_sequence.append(key)
                    api_sequences.append(mapped_api_sequence)

        except Exception as e:
            print("error in extract_api_sequence", e)
            raise

        return api_sequences

    def extract_api_call_sets(self,code_info):
        """
        Extract API call sets from the code (list)
        Extract API call sets directly from code_info
        Note: Remove redundant API calls, such as block, Model
        """
        code_flatten_infos = self.extract_code_flatten_info(code_info)
        api_call_sets = []
        for code_flatten_info in code_flatten_infos:
            for key in code_flatten_info[1]:
                if key not in api_call_sets:
                    api_call_sets.append(key)
        
        api_call_sets = [api_call for api_call in api_call_sets if 'block' not in str(api_call)]
        api_call_sets = [api_call for api_call in api_call_sets if api_call != 'Model']

        return api_call_sets

