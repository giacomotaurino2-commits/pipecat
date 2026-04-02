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

    # VAD OTTIMIZZATO (NIENTE PIU' RITARDI): 
    # min_volume=0.1 blocca il fruscio continuo (la vera causa dei 30 secondi).
    # stop_secs=0.5 permette pause naturali ma taglia e invia la frase appena hai finito.
    silero_vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.5,     
        start_secs=0.2,      
        stop_secs=0.5,
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

    # RIMOSSO IL PARAMETRO CHE MANDAVA IN CRASH IL SERVER
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
    
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            temperature=0.5, 
        )
    )

    system_prompt = """SEI L'ASSISTENTE VOCALE E CONSULENTE TECNOLOGICO DI ROJAK, una software house d'avanguardia.
    Sei dotato di un'intelligenza brillante e puoi rispondere a qualsiasi domanda tecnologica, di business o sui nostri servizi. Non sei un robot limitato a un copione.
    
    REGOLE DI CONVERSAZIONE:
    1. NATURALEZZA: Parla in modo sciolto, naturale e professionale. Rispondi con competenza a qualsiasi dubbio del cliente. NON usare MAI asterischi, cancelletti o elenchi puntati.
    2. BREVITÀ INTELLIGENTE: Sii discorsivo ma conciso. Cerca di mantenere le risposte entro 2 o 3 frasi logiche.
    3. GESTIONE DELL'IMPREVISTO: Se l'utente ti chiede cose fuori contesto, filosofiche o complesse, usa la tua intelligenza per dare una risposta sensata, poi riporta elegantemente il discorso su come Rojak può aiutarli.
    4. OBIETTIVO FINALE: Il tuo scopo ultimo è capire le esigenze del cliente e fissare una Discovery Call di 15 minuti con un nostro esperto.
    
    CHI SIAMO:
    Rojak sviluppa soluzioni AI avanzate, agenti intelligenti, CRM su misura e software custom per trasformare le aziende. I prezzi sono personalizzati."""

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

    # Interruzioni disattivate: il bot terminerà sempre le sue frasi senza bloccarsi.
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
