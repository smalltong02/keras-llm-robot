#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Task 1: Start conda environment and run Python web server
gnome-terminal -- bash -c "cd $DIR; conda activate keras-llm-robot; python __webgui_server__.py --webui; exec bash"

# Task 2: Start SSL proxy
gnome-terminal -- bash -c "cd $DIR/tools; ssl-proxy-linux -from 0.0.0.0:4480 -to 127.0.0.1:8818; exec bash"