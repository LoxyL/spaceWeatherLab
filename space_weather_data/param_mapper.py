from typing import List

class ParameterMapper:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def map(self, parameters: List[str]) -> List[str]:
        """
        Expands parameter aliases based on the mapping.
        e.g., 'B' -> ['Bx', 'By', 'Bz']
        e.g., 'V' -> ['Vsw']
        """
        mapped_params = []
        for p in parameters:
            mapped_value = self.mapping.get(p)
            
            # If the mapped value is a list, it's a vector to be expanded
            if isinstance(mapped_value, list):
                mapped_params.extend(mapped_value)
            # If it's a string, it's a simple alias
            elif isinstance(mapped_value, str):
                mapped_params.append(mapped_value)
            # Otherwise, no mapping found, use the original parameter
            else:
                mapped_params.append(p)

        return list(dict.fromkeys(mapped_params)) # Preserve order and remove duplicates
