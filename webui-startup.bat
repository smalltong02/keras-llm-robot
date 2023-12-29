@echo off

REM Task 1: Start conda environment and run Python web server
start cmd /k "cd /d %~dp0 && conda activate keras-llm-robot && python __webgui_server__.py --webui"

REM Task 2: Start SSL proxy
start cmd /k "cd /d %~dp0tools && ssl-proxy -from 0.0.0.0:4430 -to 127.0.0.1:8818"

exit