import asyncio
import os
import aiohttp

from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)
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
    async with aiohttp.ClientSession() as session:
        rest_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY", ""),
            daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
            aiohttp_session=session,
        )

        # Crea la stanza con scadenza di 1 ora
        room = await rest_helper.create_room(
            params=DailyRoomParams(
                properties=DailyRoomProperties(exp=60 * 60)
            )
        )

        # Genera il token per il bot
        token = await rest_helper.get_token(room.url, expiry_time=60 * 60)

        transport = DailyTransport(
            room.url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-2756221880dd",
        )
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )

        messages = [
            {
                "role": "system",
                "content": os.getenv(
                    "SYSTEM_PROMPT",
                    "Ciao! Sono l'assistente AI di Giacomo. Rispondo in modo naturale e molto breve.",
                ),
            }
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
