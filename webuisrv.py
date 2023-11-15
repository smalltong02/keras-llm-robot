import streamlit as st
import multiprocessing as mp
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import httpx
import os
import subprocess
from WebUI.configs import (WEBUI_SERVER)

class InnerLlmAIRobotWebUIServer():
    def __init__(self):
        super(InnerLlmAIRobotWebUIServer, self).__init__()

    def launch(self, started_event: mp.Event = None, run_mode: str = None):
        host = WEBUI_SERVER["host"]
        port = WEBUI_SERVER["port"]

        cmd = ["streamlit", "run", "webui.py",
                "--server.address", host,
                "--server.port", str(port),
                "--theme.base", "light",
                "--theme.primaryColor", "#165dff",
                "--theme.secondaryBackgroundColor", "#f5f5f5",
                "--theme.textColor", "#000000",
            ]
        if run_mode == "lite":
            cmd += [
                "--",
                "lite",
            ]
        p = subprocess.Popen(cmd)
        started_event.set()
        p.wait()