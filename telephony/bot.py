import asyncio
import os
import aiohttp
import time

from pipecat.transports.services.daily import DailyTransport, DailyParams
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


async def create_daily_room(api_key: str) -> tuple[str, str]:
    """Chiama direttamente l'API Daily senza usare DailyRESTHelper."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    exp = int(time.time()) + 3600  # scade in 1 ora

    async with aiohttp.ClientSession(headers=headers) as session:
        # 1. Crea la stanza
        async with session.post(
            "https://api.daily.co/v1/rooms",
            json={"properties": {"exp": exp}},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Errore creazione stanza Daily ({resp.status}): {text}")
            room = await resp.json()
            room_url = room["url"]
            room_name = room["name"]

        # 2. Genera il token
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            json={
                "properties": {
                    "room_name": room_name,
                    "exp": exp,
                    "is_owner": True,
                }
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Errore generazione token Daily ({resp.status}): {text}")
            token_data = await resp.json()
            token = token_data["token"]

    return room_url, token


async def main():
    api_key = os.getenv("DAILY_API_KEY", "")
    if not api_key:
        raise Exception("DAILY_API_KEY non impostata!")

    logger.info("Creazione stanza Daily...")
    room_url, token = await create_daily_room(api_key)
    logger.info(f"Stanza creata: {room_url}")

    transport = DailyTransport(
        room_url,
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
