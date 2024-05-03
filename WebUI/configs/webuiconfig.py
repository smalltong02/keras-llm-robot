import json

class InnerJsonConfigParse:
    def __init__(self, path):
        try:
            self.path = path
            self.config = None
            with open(self.path, 'r') as file:
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

class InnerJsonConfigWebUIParse(InnerJsonConfigParse):
    def __init__(self):
        super().__init__("WebUI/configs/webuiconfig.json")
    
class InnerJsonConfigPresetTempParse(InnerJsonConfigParse):
    def __init__(self):
        super().__init__("WebUI/configs/presettemplates.json")
    
class InnerJsonConfigKnowledgeBaseParse(InnerJsonConfigParse):
    def __init__(self):
        super().__init__("WebUI/configs/kbconfig.json")

class InnerJsonConfigAIGeneratorParse(InnerJsonConfigParse):
    def __init__(self):
        super().__init__("WebUI/configs/aigeneratorconfig.json")