import os
import json
import uvicorn
from loguru import logger
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response

from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import TextFrame

from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService

load_dotenv(override=True)

app = FastAPI()

@app.post("/twilio")
async def twilio_webhook(request: Request):
    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}" />
        </Connect>
    </Response>"""
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await run_bot(websocket)

async def run_bot(websocket: WebSocket):
    stream_sid = None
    call_sid = None

    async for raw_message in websocket.iter_text():
        msg = json.loads(raw_message)
        event = msg.get("event")

        if event == "connected":
            continue

        if event == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"]["callSid"]
            break

    if not stream_sid:
        return

    # VAD PERFETTO: 
    silero_vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.5,     
        start_secs=0.2,      
        stop_secs=0.2,
        min_volume=0.1       
    ))

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=silero_vad,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="it"
    )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="ee16f140-f6dc-490e-a1ed-c1d537ea0086",
            language="it"
        )
    )
    
    # DOWNGRADE AL 5.1 PER MINORE LATENZA
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-5.1",
            temperature=0.4, 
        )
    )

    # PROMPT OTTIMIZZATO PER VELOCITÀ E "RIEMPITIVI"
    system_prompt = """SEI IL CONSULENTE TECNOLOGICO DI ROJAK.
    Rispondi sempre in modo naturale, senza asterischi o elenchi. Sii conciso (2 frasi massimo).
    
    TRUCCO DI VELOCITÀ: Inizia SEMPRE le tue risposte con parole rapide come "Certo,", "Capisco,", "Assolutamente,", o "Ottima domanda,".
    
    OBIETTIVO: Rispondi alla domanda e proponi una Discovery Call di 15 minuti.
    CHI SIAMO: Sviluppiamo AI, CRM e software custom. Prezzi personalizzati."""

    greeting = "Buongiorno, grazie per aver chiamato Rojak. Sono l'assistente digitale del team, come posso aiutarla oggi?"
    
    messages = [
        {"role": "system", "content": os.getenv("SYSTEM_PROMPT", system_prompt)},
        {"role": "assistant", "content": greeting}
    ]
    
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Bot connesso")
        await task.queue_frames([TextFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Chiamata terminata")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
