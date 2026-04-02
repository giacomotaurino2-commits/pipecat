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

    # VAD BLINDATO: Standard Pipecat (0.2s) ma con min_volume a 0.5.
    # Questo significa che ignorerà TOTALMENTE respiri, fruscii di linea e rumori di fondo.
    # Ascolterà solo quando parli con un tono di voce chiaro.
    silero_vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.75,     
        start_secs=0.2,      
        stop_secs=0.2,       
        min_volume=0.5       
    ))

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=silero_vad,
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        ),
    )

    # Deepgram ottimizzato in italiano
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="it"
    )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="ee16f140-f6dc-490e-a1ed-c1d537ea0086",
        language="it"
    )
    
    # LLM impostato su gpt-4o: L'intelligenza massima disponibile per la latenza in tempo reale.
    # Temperature 0.3: Abbastanza basso da non fargli inventare nulla, ma con un pizzico di naturalezza.
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            temperature=0.3, 
        )
    )

    # PROMPT AVANZATO: Istruzioni di ferro per prevenire divagazioni e stupidità.
    system_prompt = """SEI L'ASSISTENTE VOCALE UFFICIALE DI ROJAK, una software house d'avanguardia. 
    Il tuo compito è rispondere al telefono in modo brillante, professionale ed estremamente rapido.
    
    REGOLE TASSATIVE (VIOLARLE È PROIBITO):
    1. ZERO FORMATTAZIONE: Non usare mai asterischi, cancelletti, markdown o liste. Scrivi tutto come se fosse un copione da leggere a voce.
    2. BREVITÀ ESTREMA: Le tue risposte devono essere di una o due frasi al massimo. Sii conciso.
    3. NO ALLUCINAZIONI: Se l'utente fa una domanda a cui non sai rispondere, o se capisci una frase senza senso (es. un fruscio), di' semplicemente: "Mi scusi, non ho sentito bene. Può ripetere?".
    4. IL TUO OBIETTIVO: L'unico tuo scopo è fissare una "Discovery Call" (una chiamata conoscitiva) di 15 minuti con un esperto del team.
    
    CONOSCENZA DI ROJAK: 
    - Sviluppiamo soluzioni AI (agenti intelligenti), CRM su misura e Software Custom.
    - I prezzi sono personalizzati e vengono discussi solo in call.
    
    STRUTTURA DELLA TUA RISPOSTA:
    Rispondi sempre direttamente alla domanda e chiudi immediatamente con una domanda per spingere verso la prenotazione (es. "Vuole che le fissi una chiamata con un nostro esperto?")."""

    messages = [
        {
            "role": "system",
            "content": os.getenv("SYSTEM_PROMPT", system_prompt),
        }
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

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Bot connesso")
        await task.queue_frames([TextFrame("Buongiorno, grazie per aver chiamato Rojak! Sono l'assistente digitale del team. Come posso aiutarla oggi?")])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Chiamata terminata")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
