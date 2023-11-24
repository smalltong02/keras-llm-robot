import json

class InnerJsonConfigWebUIParse:
    def __init__(self, configdata: dict):
        if configdata is None:
            return
        self.config = configdata

    def get(self, key: str) -> any:
        if self.config is None:
            return None
        value = self.config.get(key)
        return value
    
    def dump(self):
        if self.config is None:
            return None
        return self.config