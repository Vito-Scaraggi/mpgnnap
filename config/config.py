import json
from jsonschema import validate
from jsonschema.exceptions import SchemaError
from types import SimpleNamespace

class cfg():
    """
    A class for loading and validating configuration files.

    Attributes:
    - schema: A JSON schema used to validate the configuration file.
    - config: A SimpleNamespace object containing the configuration data.
    - config_json: The original configuration data in JSON format.

    Methods:
    - __init__(json_data=None): Initializes the cfg object by loading and validating the configuration file.
    - get_config(): Returns the configuration data as a SimpleNamespace object.
    - get_config_json(): Returns the configuration data in JSON format.
    """
    def __init__(self, json_data = None):
        try:
            with open("config/config_schema.json") as f:
                self.schema = json.load(f)

            if json_data is None:
                with open("config/config.json") as f:
                    json_data = json.load(f)

            validate(instance=json_data, schema=self.schema)
            self.config = json.loads( json.dumps(json_data),  object_hook= lambda x : SimpleNamespace(**x))
            self.config_json = json_data

        except FileNotFoundError:
            print("Config files not found")
        except SchemaError as e:
            print(f"Config file is not valid JSON: {str(e)}")
    
    def get_config(self):
        return self.config

    def get_config_json(self):
        return self.config_json