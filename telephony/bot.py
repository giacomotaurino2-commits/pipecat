import os
from fastapi import FastAPI, WebSocket, Request, Response
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai import OpenAIVoiceGraphiteService # Il motore Realtime
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import TextFrame

app = FastAPI()

@app.post("/twilio")
async def twilio_webhook(request: Request):
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="wss://{request.headers.get("host")}/ws" /></Connect></Response>'
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Configuriamo il trasporto per Twilio (8000Hz nativi)
    transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
        audio_out_enabled=True,
        audio_in_enabled=True,
        audio_in_sample_rate=8000,
        audio_out_sample_rate=8000,
        serializer=TwilioFrameSerializer(stream_sid=None, call_sid=None) # Si autoconfigura allo start
    ))

    # IL MOTORE TUTTO-IN-UNO (Senza Deepgram e senza Cartesia separati)
    # Questo modello sente, pensa e parla nello stesso istante.
    llm = OpenAIVoiceGraphiteService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-realtime-preview",
        voice="alloy", # Puoi scegliere tra le voci native di OpenAI che sono velocissime
    )

    sys_prompt = "Sei l'assistente Rojak. Rispondi in modo umano e veloce. Massimo 1 frase. Proponi call 15 min."
    context = LLMContext([{"role": "system", "content": sys_prompt}])

    # La pipeline ora è cortissima: Input -> Cervello -> Output
    pipeline = Pipeline([
        transport.input(),
        llm,
        transport.output()
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(t, c):
        await task.queue_frames([TextFrame("Ciao, sono l'assistente di Rojak. Dimmi tutto.")])

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
