import os
import sys
import signal
import argparse
import asyncio
import multiprocessing as mp
from WebUI.webuisrv import InnerLlmAIRobatWebUIServer

def handler(signalname):
    def f(signal_received, frame):
        raise KeyboardInterrupt(f"{signalname} received")
    return f

def main_server():
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    #if args.webui is None:
        #print("Please use the --webui parameter to start the Web UI.")
        #sys.exit(0)
    
    #webui = InnerLlmAIRobatWebUIServer()
    #webui.launch()

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_server())

    #parser = argparse.ArgumentParser(description="Process Arguments.")
    #parser.add_argument("--webui", type=str, required=True, help="Launching the Web UI interface.")

    #args = parser.parse_args()
    #main(args)