import os
import json
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
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import TextFrame, EndFrame

from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService

load_dotenv(override=True)
app = FastAPI()

SYSTEM_PROMPT = """Sei Giulia, la segretaria digitale di Rojak — un servizio di receptionist AI.
Sei professionale, rapida e diretta. Regole TASSATIVE:
- Rispondi SEMPRE in italiano
- MAX 15 parole per risposta (non di più, mai)
- Tono caldo ma efficiente, come una receptionist di lusso
- Se chiedono info sul servizio: "Rojak gestisce chiamate 24/7 con AI. Prenotiamo una Discovery Call di 15 minuti?"
- Se vogliono parlare col titolare: "Il titolare non è disponibile ora. Posso fissare una chiamata per lei?"
- Se lamentele o urgenze: "Capisco. La metto in lista prioritaria. Può lasciarmi un recapito?"
- Non fare domande multiple nella stessa risposta
- Chiudi sempre con UNA sola domanda o proposta d'azione"""

GREETING = "Buongiorno, Rojak. Giulia al telefono, come posso aiutarla?"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/twilio")
async def twilio_webhook(request: Request):
    host = request.headers.get("host")
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{host}/ws" />'
        "</Connect></Response>"
    )
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    stream_sid, call_sid = None, None

    async for raw_message in websocket.iter_text():
        msg = json.loads(raw_message)
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"]["callSid"]
            logger.info(f"Chiamata avviata | stream={stream_sid} | call={call_sid}")
            break
        if msg.get("event") == "stop":
            logger.info("Chiamata terminata prima dell'handshake")
            return

    if not stream_sid:
        logger.warning("Stream SID non ricevuto, chiudo WebSocket")
        await websocket.close()
        return

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.4))

    # ── Transport ─────────────────────────────────────────────────────────────
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=vad,
            serializer=TwilioFrameSerializer(
                stream_sid,
                call_sid,
                os.getenv("TWILIO_ACCOUNT_SID"),
                os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        ),
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-phonecall",
        language="it",
        extra={"endpointing": 200, "utterance_end_ms": 1000},
    )

    # ── TTS — nuova sintassi Settings ─────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="36d94908-c5b9-4014-b521-e69aee5bead0",
            model="sonic-multilingual",
            language="it",
            speed="normal",
        ),
    )

    # ── LLM — nuova sintassi Settings ─────────────────────────────────────────
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            max_tokens=60,
            temperature=0.4,
        ),
    )

    # ── Contesto ──────────────────────────────────────────────────────────────
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    # ── FIX PRINCIPALE: PipelineTask ora accetta solo il pipeline ─────────────
    # I parametri vanno passati come kwargs, non come oggetto PipelineParams
    task = PipelineTask(
        pipeline,
        allow_interruptions=True,
        enable_metrics=True,
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(t, c):
        logger.info("Client connesso — invio saluto")
        await task.queue_frames([TextFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(t, c):
        logger.info("Client disconnesso — chiudo pipeline")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    logger.info(f"Sessione terminata | stream={stream_sid}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level="info",
    )
