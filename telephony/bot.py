import os
import json
import time
import inspect
import uvicorn
from loguru import logger
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

import aiohttp

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import TextFrame, EndFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService

load_dotenv(override=True)
app = FastAPI()

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_API_URL = "https://api.daily.co/v1"

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


async def create_daily_room() -> dict:
    """Crea una room Daily temporanea per la chiamata."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{DAILY_API_URL}/rooms",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            json={
                "properties": {
                    "max_participants": 2,
                    "exp": int(time.time()) + 3600,
                    "enable_chat": False,
                    "enable_prejoin_ui": False,
                }
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Daily room creation failed: {text}")
            return await resp.json()


async def create_daily_token(room_url: str) -> str:
    """Crea un token owner per pipecat nella room."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{DAILY_API_URL}/meeting-tokens",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            json={
                "properties": {
                    "room_name": room_url.split("/")[-1],
                    "is_owner": True,
                    "enable_recording": False,
                }
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Daily token creation failed: {text}")
            data = await resp.json()
            return data["token"]


def build_cartesia_tts(voice_id: str):
    api_key = os.getenv("CARTESIA_API_KEY")

    try:
        if hasattr(CartesiaTTSService, "Settings"):
            settings_params = inspect.signature(CartesiaTTSService.Settings.__init__).parameters
            kwargs = {"voice": voice_id, "model": "sonic-2"}
            settings = CartesiaTTSService.Settings(**{
                k: v for k, v in kwargs.items() if k in settings_params
            })
            logger.info(f"Cartesia: Settings con params={list(kwargs.keys())}")
            return CartesiaTTSService(api_key=api_key, settings=settings)
    except Exception as e:
        logger.warning(f"Cartesia Settings fallita ({e}), provo legacy")

    try:
        init_params = inspect.signature(CartesiaTTSService.__init__).parameters
        kwargs = {"api_key": api_key, "voice_id": voice_id}
        if "model_id" in init_params:
            kwargs["model_id"] = "sonic-2"
        elif "model" in init_params:
            kwargs["model"] = "sonic-2"
        logger.info(f"Cartesia: legacy con params={list(kwargs.keys())}")
        return CartesiaTTSService(**kwargs)
    except Exception as e:
        logger.error(f"Cartesia: tutto fallito — {e}")
        raise


def build_openai_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        if hasattr(OpenAILLMService, "Settings"):
            settings_params = inspect.signature(OpenAILLMService.Settings.__init__).parameters
            kwargs = {"model": "gpt-4o-mini"}
            if "max_tokens" in settings_params:
                kwargs["max_tokens"] = 60
            if "temperature" in settings_params:
                kwargs["temperature"] = 0.4
            settings = OpenAILLMService.Settings(**{
                k: v for k, v in kwargs.items() if k in settings_params
            })
            logger.info(f"OpenAI: Settings con params={list(kwargs.keys())}")
            return OpenAILLMService(api_key=api_key, settings=settings)
    except Exception as e:
        logger.warning(f"OpenAI Settings fallita ({e}), provo legacy")

    try:
        logger.info("OpenAI: legacy minimal")
        return OpenAILLMService(api_key=api_key, model="gpt-4o-mini")
    except Exception as e:
        logger.error(f"OpenAI: tutto fallito — {e}")
        raise


def build_pipeline_task(pipeline):
    sig = inspect.signature(PipelineTask.__init__)
    params = list(sig.parameters.keys())
    logger.info(f"PipelineTask params rilevati: {params}")

    if "allow_interruptions" in params:
        return PipelineTask(pipeline, allow_interruptions=True)

    if "params" in params:
        try:
            from pipecat.pipeline.task import PipelineParams
            return PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
        except Exception as e:
            logger.warning(f"PipelineParams fallito: {e}")

    return PipelineTask(pipeline)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/twilio")
async def twilio_webhook(request: Request):
    """
    Twilio chiama questo endpoint quando arriva una telefonata.
    Creiamo una room Daily e giriamo Twilio lì dentro via SIP.
    """
    try:
        room = await create_daily_room()
        room_url = room["url"]
        room_name = room["name"]
        logger.info(f"Room Daily creata: {room_url}")

        # Avvia la pipeline in background
        import asyncio
        asyncio.create_task(run_bot(room_url, room_name))

        # Twilio si connette alla room Daily via SIP
        sip_uri = f"sip:{room_name}@sip.daily.co"
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Dial><Sip>{sip_uri}</Sip></Dial>'
            "</Response>"
        )
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        logger.error(f"Errore creazione room: {e}")
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Say language='it-IT'>Servizio temporaneamente non disponibile.</Say></Response>"
        )
        return Response(content=twiml, media_type="text/xml")


async def run_bot(room_url: str, room_name: str):
    """Pipeline pipecat con DailyTransport."""
    logger.info(f"Bot avviato per room: {room_url}")

    try:
        token = await create_daily_token(room_url)
    except Exception as e:
        logger.error(f"Errore token Daily: {e}")
        return

    # VOICE ID — sostituisci con il tuo voice_id italiano da Cartesia
    VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "36d94908-c5b9-4014-b521-e69aee5bead0")

    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.2))

    transport = DailyTransport(
        room_url,
        token,
        "Giulia",  # nome del bot nella room
        DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=vad,
            transcription_enabled=False,  # usiamo Deepgram, non Daily transcription
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-phonecall",
        language="it",
        extra={"endpointing": 200, "utterance_end_ms": 1000},
    )

    tts = build_cartesia_tts(VOICE_ID)
    llm = build_openai_llm()

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = build_pipeline_task(pipeline)

    @transport.event_handler("on_first_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"Partecipante connesso: {participant.get('id')}")
        await task.queue_frames([TextFrame(GREETING)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Partecipante uscito: {reason}")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    logger.info(f"Sessione terminata per room: {room_name}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level="info",
    )
