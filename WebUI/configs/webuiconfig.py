import json

class InnerJsonConfigWebUIParse:
    def __init__(self):
        try:
            self.config = None
            with open("WebUI/configs/webuiconfig.json", 'r') as file:
                jsondata = json.load(file)
                self.config = jsondata
        except Exception as e:
            print(e)
            return
        
    def get(self, key: str) -> any:
        if self.config is None:
            return None
        value = self.config.get(key)
        return value
    
    def dump(self):
        if self.config is None:
            return None
        return self.config
    
class InnerJsonConfigPresetTempParse:
    def __init__(self):
        try:
            self.config = None
            with open("WebUI/configs/presettemplates.json", 'r') as file:
                jsondata = json.load(file)
                self.config = jsondata
        except Exception as e:
            print(e)
            return
        
    def get(self, key: str) -> any:
        if self.config is None:
            return None
        value = self.config.get(key)
        return value
    
    def dump(self):
        if self.config is None:
            return None
        return self.config