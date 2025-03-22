import re

class ExtensionAnalyzer:

    def __init__(self, class_name, extensions):
        self.class_name = class_name
        self.extensions = extensions

    def analyze(self):
        strongest_extension = None
        max_strength = float('-inf')

        for extension in self.extensions:
            strength = self._calculate_strength(extension)

            if strength > max_strength:
                strongest_extension = extension
                max_strength = strength

        if strongest_extension:
            return f"{self.class_name}.{strongest_extension}"
        else:
            return None

    def _calculate_strength(self, extension):
        cap_count = len(re.findall(r'[A-Z]', extension))
        sm_count = len(re.findall(r'[a-z]', extension))

        return cap_count - sm_count

# Test case
class_name = "Slices"
extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']

analyzer = ExtensionAnalyzer(class_name, extensions)
strongest_extension = analyzer.analyze()

print(strongest_extension)  # Output: Slices.SErviNGSliCes