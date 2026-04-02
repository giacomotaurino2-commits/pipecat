import os
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, Response

from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams
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
    # Genera il TwiML dinamicamente usando l'host corrente
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="wss://{request.headers.get("host")}/ws" /></Connect></Response>'
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_sid, call_sid = None, None
    async for msg in websocket.iter_json():
        if msg.get("event") == "start":
            stream_sid, call_sid = msg["start"]["streamSid"], msg["start"]["callSid"]
            break

    # SETTAGGI AGGRESSIVI: stop_secs 0.2 e min_volume 0.1 per reagire subito
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.2, min_volume=0.1))
    
    transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
        vad_analyzer=vad,
        serializer=TwilioFrameSerializer(stream_sid, call_sid, os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    ))

    # DEEPGRAM PAY-AS-YOU-GO: Usiamo nova-2-phonecall con endpointing rapidissimo
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"), 
        model="nova-2-phonecall", 
        language="it"
    )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"), 
        voice_id="36d94908-c5b9-4014-b521-e69aee5bead0"
    )
    
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model="gpt-4o"
    )

    # PROMPT PER VELOCITÀ MASSIMA
    sys_prompt = "Sei l'assistente Rojak. Rispondi in massimo 12 parole. Sii naturale, cordiale e proponi sempre una Discovery Call di 15 minuti."
    
    context = LLMContext([{"role": "system", "content": sys_prompt}])
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg
    ])

    # allow_interruptions=True permette una conversazione naturale
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(t, c):
        logger.info("Chiamata agganciata - Invio saluto")
        await task.queue_frames([TextFrame("Buongiorno, sono l'assistente di Rojak. Come posso aiutarla oggi?")])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

if __name__ == "__main__":
    # Avvio standard uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
