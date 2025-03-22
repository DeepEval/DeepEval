import yaml


class GetBenchmarkYamlfiles:
    """
    A class for obtaining yaml files in the benchmark.
    """

    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path

    def read_yaml_data(self):

        file = open(self.yaml_file_path, "r", encoding="utf-8")
        file_data = file.read()
        file.close()
        task = yaml.load(file_data, Loader=yaml.FullLoader) 

        return task

