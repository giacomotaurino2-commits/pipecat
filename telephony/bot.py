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
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import TextFrame, EndFrame

from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService

load_dotenv(override=True)
app = FastAPI()

# ── Sistema Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Sei Giulia, la segretaria digitale di Rojak — un servizio di receptionist AI.
Sei professionale, rapida e diretta. Regole TASSATIVE:
- Rispondi SEMPRE in italiano
- MAX 15 parole per risposta (non di più, mai)
- Tono caldo ma efficiente, come una receptionist di lusso
- Se chiedono info sul servizio: "Rojak gestisce chiamate 24/7 con AI. Prenotiamo una Discovery Call di 15 minuti?"
- Se vogliono parlare col titolare: "Il titolare non è disponibile ora. Posso fissare una chiamata per lei?"
- Se lamentele o urgenze: "Capisco. La metto in lista prioritaria. Può lasciarmi un recapito?"
- Mai dire "Come posso aiutarla?" più di una volta
- Non fare domande multiple nella stessa risposta
- Chiudi sempre con UNA sola domanda o proposta d'azione"""

GREETING = "Buongiorno, Rojak. Giulia al telefono, come posso aiutarla?"

# ── Twilio Webhook ─────────────────────────────────────────────────────────────
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

# ── WebSocket principale ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    stream_sid, call_sid = None, None

    # Leggi lo stream SID prima di costruire la pipeline
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

    # ── VAD: equilibrio tra reattività e falsi positivi ───────────────────────
    vad = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=0.4,          # 0.4s di silenzio → fine turno (era 0.2, troppo aggressivo)
            min_volume=0.6,         # filtra rumori di fondo
        )
    )

    # ── Transport Twilio ───────────────────────────────────────────────────────
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

    # ── STT: Deepgram Nova-2 (phonecall = ottimizzato per telefonia) ───────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-phonecall",
        language="it",
        # Endpointing ridotto = trascrizione più veloce
        extra={"endpointing": 200, "utterance_end_ms": 1000},
    )

    # ── TTS: Cartesia ──────────────────────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="36d94908-c5b9-4014-b521-e69aee5bead0",
        model_id="sonic-multilingual",   # modello attuale Cartesia multilingua
        language="it",
        speed="normal",                  # "fast" se vuoi ancora più reattività
    )

    # ── LLM: GPT-4o con parametri per risposte brevi e veloci ─────────────────
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",   # più veloce di gpt-4o per turni brevi, quasi identico
        params=OpenAILLMService.InputParams(
            max_tokens=60,     # cap duro: max ~15 parole in uscita
            temperature=0.4,   # più deterministico = più coerente
        ),
    )

    # ── Contesto LLM ──────────────────────────────────────────────────────────
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,          # utente può interrompere mid-sentence
            enable_metrics=True,               # log latenze per debugging
        ),
    )

    # ── Evento: connessione stabilita → saluto immediato ──────────────────────
    @transport.event_handler("on_client_connected")
    async def on_connected(t, c):
        logger.info("Client connesso — invio saluto")
        await task.queue_frames([TextFrame(GREETING)])

    # ── Evento: disconnessione pulita ─────────────────────────────────────────
    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(t, c):
        logger.info("Client disconnesso — chiudo pipeline")
        await task.queue_frames([EndFrame()])

    # ── Avvio ─────────────────────────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    logger.info(f"Sessione terminata | stream={stream_sid}")


# ── Health check (utile per Railway/Render) ────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level="info",
    )
