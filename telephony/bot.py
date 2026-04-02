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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.frames.frames import LLMMessagesFrame, TextFrame

load_dotenv(override=True)

# Inizializza il server web
app = FastAPI()

# 1. Quando Twilio chiama il tuo link, entra qui
@app.post("/twilio")
async def twilio_webhook(request: Request):
    logger.info("Chiamata in arrivo da Twilio...")
    host = request.headers.get("host", "")
    # Diciamo a Twilio di collegarsi al nostro WebSocket sicuro
    ws_url = f"wss://{host}/ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}" />
        </Connect>
    </Response>"""
    return Response(content=twiml, media_type="text/xml")

# 2. Twilio si collega al WebSocket ed entra qui
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await run_bot(websocket)

# 3. Il TUO codice originale che gestisce l'AI
async def run_bot(websocket: WebSocket):
    stream_sid = None
    call_sid = None

    async for raw_message in websocket.iter_text():
        msg = json.loads(raw_message)
        event = msg.get("event")

        if event == "connected":
            logger.info("Twilio: connected")
            continue

        if event == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"]["callSid"]
            logger.info(f"Twilio stream avviato — stream_sid={stream_sid}, call_sid={call_sid}")
            break

    if not stream_sid:
        logger.error("Nessun stream_sid ricevuto, chiusura.")
        return

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    # Cartesia con la voce di Lorenzo
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="ee16f140-f6dc-490e-a1ed-c1d537ea0086",
    )
    
    # OpenAI aggiornato a GPT-5.1
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5.1",
    )

    messages = [
        {
            "role": "system",
            "content": os.getenv(
                "SYSTEM_PROMPT",
                "Sei l'assistente vocale di Rojak. Rispondi in modo professionale e breve.",
            ),
        }
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Bot connesso, invio messaggio di benvenuto")
        # Passiamo le istruzioni di base
        await task.queue_frames([LLMMessagesFrame(messages)])
        # Il bot prende la parola per primo salutando il cliente
        await task.queue_frames([TextFrame("Buongiorno, grazie per aver chiamato Rojak! Sono l'assistente digitale del team. Come posso aiutarla oggi?")])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Chiamata terminata")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

# Accensione del server sulla porta 8080
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Avvio server sulla porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
