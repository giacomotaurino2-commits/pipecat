import asyncio
import os
import sys

# Nuovi percorsi per la versione 0.0.108
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.transports.services.daily import DailyRestHelper
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

async def main():
    # Prendi la porta dalle variabili di Railway
    port = int(os.getenv("PORT", 8080))
    
    async with DailyRestHelper(
        os.getenv("DAILY_API_KEY"),
        os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
    ) as rest_helper:
        # Crea la stanza Daily
        room = await rest_helper.create_room()
        
        transport = DailyTransport(
            room["url"],
            room["token"],
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
            )
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-2756221880dd",
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {"role": "system", "content": os.getenv("SYSTEM_PROMPT", "Ciao! Sono l'assistente AI di Giacomo. Rispondo in modo conciso e naturale.")},
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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([context_aggregator.claim_control_frame()])

        runner = PipelineRunner()
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
