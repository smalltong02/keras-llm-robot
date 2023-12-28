from langchain.callbacks.base import BaseCallbackHandler
import azure.cognitiveservices.speech as speechsdk
import os
import io
import base64
from openai import OpenAI
import multiprocessing
from WebUI.configs.basicconfig import TMP_DIR
from WebUI.configs.serverconfig import FSCHAT_CONTROLLER
from pydub import AudioSegment
from pydub.playback import play
from WebUI.Server.llm_api import get_speech_data
from typing import List, Dict, Any
from langchain.schema.output import LLMResult

class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text=""

class StreamSpeakHandler(BaseCallbackHandler):
    def __init__(self, 
        run_place="cloud",
        provider='',
        synthesis="en-US-AriaNeural", 
        rate="+0.00%",
        subscription=None,
        region=None):
        self.initialize = False
        if subscription == None or subscription == "":
            if provider == "AzureCloud":
                subscription=os.environ.get('SPEECH_KEY')
            elif provider == "OpenAICloud":
                subscription=os.environ.get('OPENAI_API_KEY')
        if region == None or region == "":
            region=os.environ.get('SPEECH_REGION')
        self.initialize = True
        self.run_place=run_place
        self.provider=provider
        self.speech_file = str(TMP_DIR / "speech.wav")
        self.new_sentence = ""
        # Initialize the speech synthesizer
        self.synthesis=synthesis
        self.rate=rate
        if provider == "OpenAICloud":
            if subscription is not None and subscription != "" and region is not None and region != "":
                self.speech_synthesizer = self.azure_settings(synthesis, subscription, region)

    def azure_settings(self, synthesis, subscription, region):
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription, 
            region=region
        )
        audio_output_config = speechsdk.audio.AudioOutputConfig(filename=self.speech_file)
        
        speech_config.speech_synthesis_voice_name=synthesis

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
        return speech_synthesizer

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.initialize is True:
            token_list = token.split('\n')
            if len(token_list) != 0:
                self.new_sentence += token_list[0]
            # Check if the new token forms a sentence.
            if len(token_list) != 1:
            #if token in ".:!?。：！？\n":
                # Synthesize the new sentence
                speak_this = self.new_sentence
                if self.run_place =="cloud":
                    self.speak_streamlit_cloud(speak_this)
                else:
                    self.speak_ssml_async(speak_this)
                self.new_sentence = ""
                if len(token_list) != 0:
                    self.new_sentence = "".join(token_list[1:])

    def on_llm_end(self, response, **kwargs) -> None:
        if self.initialize is True:
            if len(self.new_sentence):
                speak_this = self.new_sentence
                if self.run_place =="cloud":
                    self.speak_streamlit_cloud(speak_this)
                else:
                    self.speak_ssml_async(speak_this)
                self.new_sentence = ""

    def speak_ssml_async(self, text):
        if self.initialize is True:
            controller_address = "http://" + FSCHAT_CONTROLLER["host"] + ":" + str(FSCHAT_CONTROLLER["port"])
            r = get_speech_data(text, controller_address, self.synthesis)
            if r["code"] == 200:
                channels = r["channels"]
                sample_width = r["sample_width"]
                frame_rate = r["frame_rate"]
                decoded_data = base64.b64decode(r["speech_data"])
                audio_segment = AudioSegment(
                    decoded_data,
                    frame_rate=frame_rate,
                    sample_width=sample_width,
                    channels=channels)
                play(audio_segment)
    def speak_streamlit_cloud(self,text):
        if self.initialize is True:
            if self.provider == "AzureCloud":
                ssml_text=f"""<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" version="1.0" xml:lang="en-US">
                                <voice name="{self.synthesis}">
                                <prosody rate="{self.rate}">
                                        {text}
                                </prosody>
                                </voice>
                            </speak>"""
                speech_synthesis_result = self.speech_synthesizer.speak_ssml_async(ssml_text).get()
                if speech_synthesis_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print(f'Error synthesizing speech: {speech_synthesis_result.reason}')
                else:
                    print("speech_synthesis_result.audio_data: ", len(speech_synthesis_result.audio_data))
                    self.new_sentence = ""
                    audio_stream = speech_synthesis_result.audio_data
                    if len(audio_stream):
                        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_stream))
                        play(audio_segment)
            elif self.provider == "OpenAICloud":
                client = OpenAI()
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=self.synthesis,
                    input=text
                    )
                response.stream_to_file(self.speech_file)
                audio_segment = AudioSegment.from_mp3(file=self.speech_file)
                play(audio_segment)
                    
class LlamacppStreamCallbackHandler(BaseCallbackHandler):

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = multiprocessing.Queue()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        pass

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    # TODO implement the other methods

    def get_tokens(self) -> str:
        while not self.queue.empty():
            return self.queue.get()
        return None