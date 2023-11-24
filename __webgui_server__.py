import os
import sys
import time
import signal
import argparse
import asyncio
import platform
import langchain
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Process
from WebUI.Server.llm_api_stale import (LOG_PATH)
from WebUI.Server.utils import (set_httpx_config, get_model_worker_config, get_httpx_client, 
                                FastAPI, MakeFastAPIOffline, fschat_controller_address,
                                fschat_model_worker_address)
from __about__ import __title__, __summary__, __version__, __author__, __email__, __license__, __copyright__
from webuisrv import InnerLlmAIRobotWebUIServer
from WebUI.configs.modelconfig import (LLM_MODELS)
from WebUI.configs.serverconfig import (FSCHAT_MODEL_WORKERS, FSCHAT_CONTROLLER, HTTPX_DEFAULT_TIMEOUT,
                                        FSCHAT_OPENAI_API, API_SERVER)
from typing import Tuple, List, Dict

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--webui",
        action="store_true",
        help="run webui servers.",
        dest="webui",
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="specify model name for model worker. "
             "add addition names with space seperated to start multiple model workers.",
        dest="model_name",
    )
    args = parser.parse_args()
    return args, parser

def dump_server_info(after_start=False, args=None):
    print("\n")
    print("=" * 30 + f"{__title__} Configuration" + "=" * 30)
    print(f"OS: {platform.platform()}.")
    print(f"python: {sys.version}")
    print(f"langchain: {langchain.__version__}.")
    print(f"Version: {__version__}")
    print(f"Summary: {__summary__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print(f"License: {__license__}")
    print(f"Copyright: {__copyright__}")

def handler(signalname):
    def f(signal_received, frame):
        raise KeyboardInterrupt(f"{signalname} received")
    return f

def create_controller_app(
        dispatch_method: str,
    ) -> FastAPI:
        import fastchat.constants
        fastchat.constants.LOGDIR = LOG_PATH
        from fastchat.serve.controller import app, Controller

        controller = Controller(dispatch_method)
        sys.modules["fastchat.serve.controller"].controller = controller

        MakeFastAPIOffline(app)
        app.title = "FastChat Controller"
        app._controller = controller
        return app

def run_webui(started_event: mp.Event = None, run_mode: str = None):
    set_httpx_config()
    webui = InnerLlmAIRobotWebUIServer()
    webui.launch(started_event, run_mode)

def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")
    async def on_startup():
        if started_event is not None:
            started_event.set()

def run_controller(started_event: mp.Event = None):
    import uvicorn
    from fastapi import Body
    import time
    import sys
    set_httpx_config()

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
    )
    _set_app_event(app, started_event)

    # add interface to release and load model worker
    @app.post("/release_worker")
    def release_worker(
            model_name: str = Body(..., description="Unload the model", samples=["chatglm-6b"]),
            # worker_address: str = Body(None, description="Unload the model address", samples=[FSCHAT_CONTROLLER_address()]),
            new_model_name: str = Body(None, description="New model"),
            keep_origin: bool = Body(False, description="Second model")
    ) -> Dict:
        available_models = app._controller.list_models()
        if new_model_name in available_models:
            msg = f"The model {new_model_name} has been loaded."
            print(msg)
            return {"code": 500, "msg": msg}

        if new_model_name:
            print(f"Change model: from {model_name} to {new_model_name}")
        else:
            print(f"Stoping model: {model_name}")

        #if model_name not in available_models:
        #    msg = f"the model {model_name} is not available"
        #    print(msg)
        #    return {"code": 500, "msg": msg}

        if model_name != "":
            worker_address = app._controller.get_worker_address(model_name)
            if not worker_address:
                msg = f"can not find model_worker address for {model_name}"
                print(msg)
                return {"code": 500, "msg": msg}
        else:
            workerconfig = get_model_worker_config(model_name)
            worker_address = "http://" + workerconfig["host"] + ":" + str(workerconfig["port"])

        with get_httpx_client() as client:
            r = client.post(worker_address + "/release",
                        json={"new_model_name": new_model_name, "keep_origin": keep_origin})
            if r.status_code != 200:
                msg = f"failed to release model: {model_name}"
                print(msg)
                return {"code": 500, "msg": msg}

        if new_model_name:
            timer = HTTPX_DEFAULT_TIMEOUT  # wait for new model_worker register
            while timer > 0:
                models = app._controller.list_models()
                if new_model_name in models:
                    break
                time.sleep(1)
                timer -= 1
            if timer > 0:
                msg = f"sucess change model from {model_name} to {new_model_name}"
                print(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"failed change model from {model_name} to {new_model_name}"
                print(msg)
                return {"code": 500, "msg": msg}
        else:
            timer = HTTPX_DEFAULT_TIMEOUT  # wait for new model_worker register
            while timer > 0:
                models = app._controller.list_models()
                if model_name not in models:
                    break
                time.sleep(1)
                timer -= 1
                app._controller.refresh_all_workers()
            if timer > 0:
                msg = f"success to release model: {model_name}"
                print(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"failed to release model: {model_name}"
                print(msg)
                return {"code": 500, "msg": msg}

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    uvicorn.run(app, host=host, port=port)

def create_empty_worker_app() -> FastAPI:
    from fastchat.serve.base_model_worker import app
    MakeFastAPIOffline(app)
    app.title = f"FastChat empty Model"
    app._worker = ""
    return app

def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    import fastchat.constants
    from fastchat.serve.base_model_worker import app
    fastchat.constants.LOGDIR = LOG_PATH
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)
    if worker_class := kwargs.get("langchain_model"):
        worker = ""
    # Online model
    elif worker_class := kwargs.get("worker_class"):
        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)

    # Local model
    else:
        #from WebUI.configs.modelconfig import VLLM_MODEL_DICT
        from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id

        args.gpus = "0"
        args.max_gpu_memory = "22GiB"
        args.num_gpus = 1

        args.load_8bit = False
        args.cpu_offloading = None
        args.gptq_ckpt = None
        args.gptq_wbits = 16
        args.gptq_groupsize = -1
        args.gptq_act_order = False
        args.awq_ckpt = None
        args.awq_wbits = 16
        args.awq_groupsize = -1
        args.model_names = [""]
        args.conv_template = None
        args.limit_worker_concurrency = 5
        args.stream_interval = 2
        args.no_register = False
        args.embed_in_truncate = False
        for k, v in kwargs.items():
            setattr(args, k, v)
        if args.gpus:
            if args.num_gpus is None:
                args.num_gpus = len(args.gpus.split(','))
            if len(args.gpus.split(",")) < args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        gptq_config = GptqConfig(
            ckpt=args.gptq_ckpt or args.model_path,
            wbits=args.gptq_wbits,
            groupsize=args.gptq_groupsize,
            act_order=args.gptq_act_order,
        )
        awq_config = AWQConfig(
            ckpt=args.awq_ckpt or args.model_path,
            wbits=args.awq_wbits,
            groupsize=args.awq_groupsize,
        )

        worker = ModelWorker(
            controller_addr=args.controller_address,
            worker_addr=args.worker_address,
            worker_id=worker_id,
            model_path=args.model_path,
            model_names=args.model_names,
            limit_worker_concurrency=args.limit_worker_concurrency,
            no_register=args.no_register,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            stream_interval=args.stream_interval,
            conv_template=args.conv_template,
            embed_in_truncate=args.embed_in_truncate,
        )
        sys.modules["fastchat.serve.model_worker"].args = args
        sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
        # sys.modules["fastchat.serve.model_worker"].worker = worker
        sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app

def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
    ) -> FastAPI:
        import fastchat.constants
        fastchat.constants.LOGDIR = LOG_PATH
        from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
        from fastchat.utils import build_logger
        logger = build_logger("openai_api", "openai_api.log")
        logger.setLevel(log_level)

        app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        sys.modules["fastchat.serve.openai_api_server"].logger = logger
        app_settings.controller_address = controller_address
        app_settings.api_keys = api_keys

        MakeFastAPIOffline(app)
        app.title = "FastChat OpeanAI API Server"
        return app

def run_openai_api(started_event: mp.Event = None):
    import uvicorn
    import sys
    set_httpx_config()

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr, log_level="INFO")  # TODO: not support keys yet.
    _set_app_event(app, started_event)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    uvicorn.run(app, host=host, port=port)

def run_model_worker(
        model_name: str = LLM_MODELS[0],
        controller_address: str = "",
        q: mp.Queue = None,
        started_event: mp.Event = None,
    ):
    import uvicorn
    from fastapi import Body
    import sys
    set_httpx_config()

    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    if model_name == "":
        app = create_empty_worker_app()
    else:
        app = create_model_worker_app(log_level="INFO", **kwargs)

    _set_app_event(app, started_event)
    # add interface to release and load model
    @app.post("/release")
    def release_model(
        new_model_name: str = Body(None, description="Load new Model"),
        keep_origin: bool = Body(False, description="keep origin Model and Load new Model")
    ) -> Dict:
        if keep_origin:
            if new_model_name:
                q.put([model_name, "start", new_model_name])
        else:
            if new_model_name:
                q.put([model_name, "replace", new_model_name])
            else:
                q.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}

    uvicorn.run(app, host=host, port=port)

def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    from WebUI.Server.api import create_app
    import uvicorn
    set_httpx_config()

    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = API_SERVER["host"]
    port = API_SERVER["port"]

    uvicorn.run(app, host=host, port=port)

def main_server():
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    queue = manager.Queue()
    args, parser = parse_args()

    if args.webui is None:
        print("Please use the --webui parameter to start the Web GUI for AI Robot.")
        sys.exit(0)

    args.openai_api = True
    args.model_worker = True
    args.api = True
    args.api_worker = True
    run_mode = None
    
    dump_server_info(args=args)

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2
    
    controller_started = manager.Event()
    if args.openai_api:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WebUI Server",
            kwargs=dict(started_event=webui_started, run_mode = run_mode),
            daemon=True,
        )
        processes["webui"] = process

    model_worker_started = []
    if args.model_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if not config.get("online_api"):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"model_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["model_worker"][model_name] = process

    if args.api_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if (config.get("online_api")
                and config.get("worker_class")
                and model_name in FSCHAT_MODEL_WORKERS):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["online_api"][model_name] = process

    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        processes["api"] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            if p:= processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait()

            if p:= processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for e in model_worker_started:
                e.wait()

            if p:= processes.get("api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait()

            if p:= processes.get("webui"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait()

            while True:
                cmd = queue.get()
                e = manager.Event()
                if isinstance(cmd, list):
                    model_name, cmd, new_model_name = cmd
                    if cmd == "start":
                        print(f"Change to new model: {new_model_name}")
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {new_model_name}",
                            kwargs=dict(model_name=new_model_name,
                                        controller_address=args.controller_address,
                                        q=queue,
                                        started_event=e),
                            daemon=True,
                        )
                        process.start()
                        process.name = f"{process.name} ({process.pid})"
                        processes["model_worker"][new_model_name] = process
                        e.wait()
                        print(f"The model: {new_model_name} running!")
                    elif cmd == "stop":
                        if process := processes["model_worker"].pop(model_name):
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            print(f"Stop model: {model_name}")
                            print(f"Start empty model!")
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - None",
                                kwargs=dict(model_name="",
                                            controller_address=args.controller_address,
                                            q=queue,
                                            started_event=e),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][""] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            print(f"Loading None Model, used: {timing}.")
                        else:
                            print(f"Can not find the model: {model_name}")
                    elif cmd == "replace":
                        if process := processes["model_worker"].pop(model_name, None):
                            print(f"Stop model: {model_name}")
                            start_time = datetime.now()
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - {new_model_name}",
                                kwargs=dict(model_name=new_model_name,
                                            controller_address=args.controller_address,
                                            q=queue,
                                            started_event=e),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][new_model_name] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            print(f"Loading new model: {new_model_name}. used: {timing}.")
                        else:
                            print(f"Can not find the model: {model_name}")

        except Exception as e:
            print("Caught KeyboardInterrupt! Setting stop event...")
        finally:
            for p in processes.values():
                print("Sending SIGKILL to %s", p)
                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                print("Process status: %s", p)

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
