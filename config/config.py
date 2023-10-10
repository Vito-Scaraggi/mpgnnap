import json
from jsonschema import validate
from jsonschema.exceptions import SchemaError
from types import SimpleNamespace

class cfg():
    def __init__(self):
        try:
            with open("config/config_schema.json") as f:
                self.schema = json.load(f)

            with open("config/config.json") as f:
                config = json.load(f)
                validate(instance=config, schema=self.schema)
                self.config = json.loads( json.dumps(config),  object_hook= lambda x : SimpleNamespace(**x))
                self.config_json = config

        except FileNotFoundError:
            print("Config files not found")
        except SchemaError as e:
            print(f"Config file is not valid JSON: {str(e)}")
    
    def get_config(self):
        return self.config

    def get_config_json(self):
        return self.config_json