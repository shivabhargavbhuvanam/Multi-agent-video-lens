import os
from llama_index.core import Settings
from pinecone import Pinecone
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Event
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core import StorageContext
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from colorama import Fore, Style
from dotenv import load_dotenv
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

load_dotenv()

# ── Settings ───────────────────────────────────────────────────────────────────
Settings.llm = OpenAI(
    model="gpt-4", temperature=0.1,
    api_key=os.environ.get("OPENAI_API_KEY", "")
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ── Pinecone ───────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
_index = pc.Index("video-analysis-index-v2")

image_captioning_vector_store = PineconeVectorStore(
    pinecone_index=_index, namespace="Default")
transcripts_vector_store = PineconeVectorStore(
    pinecone_index=_index, namespace="Default")
yolo_vector_store = PineconeVectorStore(
    pinecone_index=_index, namespace="Default")

image_captioning_index = VectorStoreIndex.from_vector_store(
    image_captioning_vector_store)
transcripts_index = VectorStoreIndex.from_vector_store(
    transcripts_vector_store)
yolo_index = VectorStoreIndex.from_vector_store(yolo_vector_store)

# ── Prompt ─────────────────────────────────────────────────────────────────────
qa_prompt = PromptTemplate(
    "Context information is below, including timestamps.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, including timestamps, and not prior knowledge, "
    "answer the query and return the exact timestamp where the event occurs.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# ── RAG Query Engine ───────────────────────────────────────────────────────────


class RAGStringQueryEngine(CustomQueryEngine):
    """Calls Pinecone retriever then GPT-4 to answer with timestamps."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str) -> str:
        nodes = self.retriever.retrieve(query_str)

        if not nodes:
            print("No relevant nodes found in Pinecone for this query.")
            return "No relevant data found."

        context_str = "\n\n".join([
            f"Content: {n.node.get_content()}\nTimestamp: {n.node.metadata.get('timestamp', 'N/A')}"
            for n in nodes
        ])

        print(f"\n--- Retrieved {len(nodes)} nodes from Pinecone ---")
        for n in nodes:
            print(
                f"  Timestamp: {n.node.metadata.get('timestamp', 'N/A')} | Agent: {n.node.metadata.get('agent', 'N/A')}")
        print("---\n")

        try:
            response = self.llm.complete(
                "You are a video analysis assistant. "
                "Based ONLY on the context below, answer the user's question "
                "and include the relevant timestamp(s) in your answer.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query_str}\n\n"
                "Answer (include the timestamp seconds):"
            )
            result = str(response)
            print("GPT-4 Response: " + result)
            return result
        except Exception as e:
            print("GPT-4 call failed:", e)
            return "FAILED"


gpt = OpenAI(model="gpt-4", temperature=0.7,
             api_key=os.environ.get("OPENAI_API_KEY", ""))

# ── Metadata filters ───────────────────────────────────────────────────────────
filters_image_caption = MetadataFilters(filters=[
    MetadataFilter(key="agent", operator=FilterOperator.EQ,
                   value="image_captioning"),
])
filters_transcripts = MetadataFilters(filters=[
    MetadataFilter(key="agent", operator=FilterOperator.EQ,
                   value="transcripts"),
])
filters_yolo = MetadataFilters(filters=[
    MetadataFilter(key="agent", operator=FilterOperator.EQ, value="yolo"),
])

image_captioning_retriever = image_captioning_index.as_retriever(
    filters=filters_image_caption)
transcripts_retriever = transcripts_index.as_retriever(
    filters=filters_transcripts)
yolo_retriever = yolo_index.as_retriever(filters=filters_yolo)

synthesizer = get_response_synthesizer(response_mode="compact")

image_captioning_engine = RAGStringQueryEngine(
    retriever=image_captioning_retriever,
    response_synthesizer=synthesizer, llm=gpt, qa_prompt=qa_prompt,
)
transcripts_engine = RAGStringQueryEngine(
    retriever=transcripts_retriever,
    response_synthesizer=synthesizer, llm=gpt, qa_prompt=qa_prompt,
)
yolo_engine = RAGStringQueryEngine(
    retriever=yolo_retriever,
    response_synthesizer=synthesizer, llm=gpt, qa_prompt=qa_prompt,
)

# ── Events ─────────────────────────────────────────────────────────────────────


class OrchestratorEvent(Event):
    request: str


class ImageCaptioningEvent(Event):
    request: str


class TranscriptsEvent(Event):
    request: str


class YoloEvent(Event):
    request: str


# ── Workflow ───────────────────────────────────────────────────────────────────
class MultiAgentWorkflow(Workflow):
    """
    Multi-agent workflow.

      StartEvent → start → OrchestratorEvent
      OrchestratorEvent → orchestrator → [ImageCaptioningEvent | TranscriptsEvent | YoloEvent | StopEvent]
      Specialized step → RAG query → _ask_next → OrchestratorEvent | StopEvent

    Routing functions write into a shared `decision` dict.
    The `if decision["route"] is None` guard prevents double-calls from
    overwriting the first (correct) routing decision.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = OpenAI(
            model="gpt-4", temperature=0.4,
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
        self._tried_tools: List[str] = []

    # ── Entry ──────────────────────────────────────────────────────────────────
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> OrchestratorEvent:
        print(Fore.CYAN + "\nHello! I can help you query the video. Ask me anything about:" + Style.RESET_ALL)
        print("  • What happens visually in scenes  (image captions)")
        print("  • What is being said               (transcripts)")
        print("  • What objects appear on screen    (YOLO detection)\n")
        user_input = input("> ").strip()
        return OrchestratorEvent(request=user_input)

    # ── Orchestrator ───────────────────────────────────────────────────────────
    @step
    async def orchestrator(
        self, ctx: Context, ev: OrchestratorEvent
    ) -> ImageCaptioningEvent | TranscriptsEvent | YoloEvent | StopEvent:
        """
        GPT-4 calls one routing function which writes into `decision`.
        The `if decision["route"] is None` guard on every routing function
        ensures that if GPT-4 calls two functions (despite allow_parallel=False),
        only the first call's decision is kept.
        After agent.chat() returns, we read `decision` and return the correct event.
        """
        print(f"\nOrchestrator received: {ev.request}")

        tried = self._tried_tools
        decision = {"route": None}  # first routing call wins

        # ── routing functions ──────────────────────────────────────────────────
        def route_image_captioning() -> str:
            """Use for colours, clothing, food, scenes, people in the video, who is present, what someone is holding or doing, visual appearance, atmosphere, timestamps of actions."""
            if decision["route"] is None:
                decision["route"] = "image_captioning"
                self._tried_tools.append("route_image_captioning")
                print("→ Image Captioning Agent selected")
            return "ok"

        def route_transcripts() -> str:
            """Use for speech, dialogue, audio, what was said, names mentioned, language spoken."""
            if decision["route"] is None:
                decision["route"] = "transcripts"
                self._tried_tools.append("route_transcripts")
                print("→ Transcripts Agent selected")
            return "ok"

        def route_yolo() -> str:
            """Use ONLY when the user explicitly asks about object detection or detected objects. Do NOT use for colour, food, people, or general visual questions."""
            if decision["route"] is None:
                decision["route"] = "yolo"
                self._tried_tools.append("route_yolo")
                print("→ YOLO Agent selected")
            return "ok"

        def route_stop() -> str:
            """Use when the user wants to exit, quit, or says goodbye."""
            if decision["route"] is None:
                decision["route"] = "stop"
                print("→ Stop selected")
            return "ok"

        # Build tool list — exclude already-tried routes this turn
        tools = []
        if "route_image_captioning" not in tried:
            tools.append(FunctionTool.from_defaults(
                fn=route_image_captioning, name="route_image_captioning",
                description="Use for colours, clothing, food, scenes, people in the video, who is present, what someone is holding or doing, visual appearance, atmosphere, timestamps of actions."
            ))
        if "route_transcripts" not in tried:
            tools.append(FunctionTool.from_defaults(
                fn=route_transcripts, name="route_transcripts",
                description="Use for speech, dialogue, audio content, what was said, names mentioned."
            ))
        if "route_yolo" not in tried:
            tools.append(FunctionTool.from_defaults(
                fn=route_yolo, name="route_yolo",
                description="Use ONLY when user explicitly asks about object detection or detected objects."
            ))
        tools.append(FunctionTool.from_defaults(
            fn=route_stop, name="route_stop",
            description="Use when the user wants to exit or says goodbye."
        ))

        system_prompt = (
            "You are a routing agent for a video analysis system.\n"
            "Call exactly ONE routing function based on the user's question. "
            "Do NOT answer the question yourself — just pick the right route.\n\n"
            "Routing rules (follow strictly):\n"
            "- route_transcripts: speech, dialogue, audio, what was said, names mentioned, language spoken\n"
            "- route_image_captioning: colours, clothing, food visible, scenes, what things look like, "
            "people in the video, who is present, what someone is holding or doing, "
            "people's appearance, atmosphere, what is happening, timestamps of actions\n"
            "- route_yolo: ONLY when user explicitly asks about object detection or detected objects. "
            "Do NOT use for colour, food, people, or general visual questions.\n"
            "- route_stop: user wants to exit or says goodbye\n\n"
            "Already tried this turn: " +
            (", ".join(tried) if tried else "none")
        )

        worker = FunctionCallingAgentWorker.from_tools(
            tools=tools, llm=self._llm,
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt,
        )
        worker.as_agent().chat(ev.request)  # decision gets set inside routing fn

        # Return event directly — first routing call wins
        route = decision["route"]
        if route == "image_captioning":
            return ImageCaptioningEvent(request=ev.request)
        elif route == "transcripts":
            return TranscriptsEvent(request=ev.request)
        elif route == "yolo":
            return YoloEvent(request=ev.request)
        else:
            return StopEvent()

    # ── Image Captioning ───────────────────────────────────────────────────────
    @step
    async def image_captioning(
        self, ctx: Context, ev: ImageCaptioningEvent
    ) -> OrchestratorEvent | StopEvent:
        print(f"\n[Image Captioning Agent] Query: {ev.request}")
        result = image_captioning_engine.query(ev.request)
        print(Fore.MAGENTA + f"\nAnswer: {result}" + Style.RESET_ALL)
        return self._ask_next()

    # ── Transcripts ────────────────────────────────────────────────────────────
    @step
    async def transcripts(
        self, ctx: Context, ev: TranscriptsEvent
    ) -> OrchestratorEvent | StopEvent:
        print(f"\n[Transcripts Agent] Query: {ev.request}")
        result = transcripts_engine.query(ev.request)
        print(Fore.MAGENTA + f"\nAnswer: {result}" + Style.RESET_ALL)
        return self._ask_next()

    # ── YOLO ───────────────────────────────────────────────────────────────────
    @step
    async def yolo(
        self, ctx: Context, ev: YoloEvent
    ) -> OrchestratorEvent | StopEvent:
        print(f"\n[YOLO Agent] Query: {ev.request}")
        result = yolo_engine.query(ev.request)
        print(Fore.MAGENTA + f"\nAnswer: {result}" + Style.RESET_ALL)
        return self._ask_next()

    # ── Next-question helper ───────────────────────────────────────────────────
    def _ask_next(self) -> OrchestratorEvent | StopEvent:
        print()
        user_input = input(
            "Ask another question (or 'exit' to quit):\n> ").strip()
        if user_input.lower() in ("exit", "quit", "q", "bye", "stop"):
            return StopEvent()
        self._tried_tools = []  # reset for fresh routing next question
        return OrchestratorEvent(request=user_input)


# ── Diagram ────────────────────────────────────────────────────────────────────
draw_all_possible_flows(
    MultiAgentWorkflow, filename="multi-agent-workflow.html")


# ── Run ────────────────────────────────────────────────────────────────────────
async def main():
    workflow = MultiAgentWorkflow(timeout=1200, verbose=True)
    result = await workflow.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
