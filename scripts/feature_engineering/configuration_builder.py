import yaml
import copy

class PipelineConfigurator:
    def __init__(self, base_yaml_path: str):
        with open(base_yaml_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            self.handler_kwargs = self.config.get("data_handler", {}).get("kwargs", {})

    def insert_processor_after(self, target_processor_name: str, new_processor_config: dict):
        """
        insert a new processor after the target processor in both learn_processors and infer_processors
        dynamically adjust the processor chain during different model's feature engineering
        """
        phases = ["learn_processors", "infer_processors"]
        for phase in phases:
            processors_list = self.handler_kwargs.get(phase, [])
            insert_idx = -1

            for i, proc in enumerate(processors_list):
                if proc.get("class") == target_processor_name:
                    insert_idx = i
                    break

            if insert_idx >= 0:
                processors_list.insert(insert_idx + 1, copy.deepcopy(new_processor_config))
            else:
                processors_list.append(copy.deepcopy(new_processor_config))

        return self

    def build(self) -> dict:
        return self.config.get("data_handler")
