import json
import time
from websockets.sync.client import connect
import WebUI.Server.interpreter_wrapper.terminal.status_code as status_code

KERAS_INTERPRETER_TERMINAL_WIN = "keras-terminal.exe"
KERAS_INTERPRETER_TERMINAL_DARWIN = "keras-terminal-darwin"
KERAS_INTERPRETER_TERMINAL_LINUX = "keras-terminal-ubuntu"

class BaseTerminal:
    def __init__(self):
        pass

    def chat(self, query):
        pass

class Terminal(BaseTerminal):
    def __init__(self, 
            host : str = "localhost",
            port : int = 20020,
            docker_mode : bool = False,
            ):
        self.languages = []
        self.host = host
        self.port = port
        self.terminal_url = f"ws://{self.host}:{self.port}"
        self.docker_mode = docker_mode
        self.valid = False
        if self.keep_alive_terminal(0) is False:
            if self.docker_mode:
                self.valid = self.start_docker_terminal()
            else:
                self.valid = self.start_local_terminal()
        self.config = None
        if self.valid:
             self.config = self.get_terminal_config()

    def get_languages(self):
         if self.config is not None:
             return self.config["languages"]
         return []

    def keep_alive_terminal(self, max_retries) -> bool:
            retry_count = 0
            url = self.terminal_url + "/test"
            while retry_count <= max_retries:
                try:
                    with connect(url) as websocket:
                        websocket.send(json.dumps({"message": "hello"}))
                        response = websocket.recv()
                        print("keep alive: ", response)
                        self.valid = True
                        return True
                except Exception as _:
                    pass
                time.sleep(1)
                retry_count += 1
            return False
    
    def start_local_terminal(self):
            import sys
            import subprocess
            local_command = ""
            if sys.platform.startswith("win32"):
                local_command = [f"./tools/{KERAS_INTERPRETER_TERMINAL_WIN}", "--port", f"{self.port}"]
            elif sys.platform.startswith("linux"):
                 local_command = [f"./tools/{KERAS_INTERPRETER_TERMINAL_LINUX}", "--port", f"{self.port}"]
            elif sys.platform.startswith("darwin"):
                local_command = [f"./tools/{KERAS_INTERPRETER_TERMINAL_DARWIN}", "--port", f"{self.port}"]
            else:
                 return False
            try:
                subprocess.Popen(local_command)
                time.sleep(1)
                return self.keep_alive_terminal(20)
            except Exception as e:
                print(e)
            return False

    def start_docker_terminal(self):
        import subprocess
        docker_command = ["docker", "run", "-d", "-p", f"{self.port}:{self.port}", "smalltong02/keras-interpreter-terminal"]
        try:
            subprocess.Popen(docker_command)
            time.sleep(1)
            return self.keep_alive_terminal(3)
        except Exception as _:
            pass
        return False
    
    def get_terminal_config(self):
        url = self.terminal_url + "/get_config"
        try:
            with connect(url) as websocket:
                websocket.send("get_config")
                response = websocket.recv()
                response = json.loads(response)
                if response["status_code"] <= 300:
                    config = response["content"]
                    return json.loads(config)
        except Exception as _:
                    pass
        return None

    def run(self, language, code):
        url = self.terminal_url + "/run_code"
        try:
            with connect(url) as websocket:
                send_data = {'language': language ,'code': code}
                websocket.send(json.dumps(send_data))
                while True:
                    response = websocket.recv()
                    response = json.loads(response)
                    if response["status_code"] <= 300:
                        if response["format"] == "text":
                            answer = response["content"]
                            print(f"received data: {answer}")
                            if response["status_code"] == status_code.STATUS_CODE_TRUNK:
                                yield answer
                                continue
                            break
                    else:
                        answer = response["content"]
                        print(f"received error: {answer}")
                        break
        except Exception as _:
                    pass
        return None