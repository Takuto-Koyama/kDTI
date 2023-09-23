import json 

class Config:
    def __init__(self, json_file_path):
        
        self._set_default()
        
        with open(json_file_path, "r") as f:
            self.config = json.load(f)
        
        self._set_config()
        self._output_config()
    
    def _set_default(self):
        self.input_csv = "../data/sample.csv"
        self.output_dir = "../data/"
        self.mode = "train"
        
                    
    def _set_config(self):
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def _output_config(self):
        print("======== Configurations ========")
        for key, value in self.config.items():
            print("{}: {}".format(key, value))
        print("================================")
    