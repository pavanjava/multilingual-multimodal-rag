from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from tools.rag_tools import retrieve_text_from_store, retrieve_image_from_store, ingest_to_vec_db, create_embeddings

import os
import json


class MultilingualMultimodalRAG:
    def __init__(self):
        # Initialize the llm model
        self.llm = Anthropic(model="claude-3-5-sonnet-20241022", temperature=0.2)

    async def invoke_ingest_workflow(self):
        embed_agent = FunctionAgent(
            name="embeddings creator",
            description="Performs embeddings creation task",
            system_prompt="You are a assistant to create embeddings for the given data",
            tools=[
                FunctionTool.from_defaults(fn=create_embeddings),
            ],
            llm=self.llm,
            can_handoff_to=["data ingestor"]
        )

        ingest_agent = FunctionAgent(
            name="data ingestor",
            description="Performs data ingestion task",
            system_prompt="You are a assistant to ingest data into vector database",
            tools=[
                FunctionTool.from_defaults(fn=ingest_to_vec_db),
            ],
            llm=self.llm
        )

        # Create and run the workflow
        workflow = AgentWorkflow(
            agents=[embed_agent, ingest_agent], root_agent="embeddings creator", timeout=300
        )

        await workflow.run(user_msg="embed the data and then ingest it to vector database")

    async def invoke_text2img_rag_workflow(self):
        retrieval_agent = FunctionAgent(
            name="retrieval agent",
            description="Performs retrieval for the given user query",
            system_prompt="You are an assistant to perform retrival for the given user query",
            tools=[
                FunctionTool.from_defaults(fn=retrieve_image_from_store)
            ],
            llm=self.llm
        )
        # Create and run the workflow
        workflow = AgentWorkflow(
            agents=[retrieval_agent], root_agent="retrieval agent", timeout=300
        )

        await workflow.run(user_msg="user interacting with Spring AI system")

    async def invoke_img2text_rag_workflow(self):
        retrieval_agent = FunctionAgent(
            name="retrieval agent",
            description="Performs retrieval for the given user query",
            system_prompt="You are an assistant to perform retrival for the given user query",
            tools=[
                FunctionTool.from_defaults(fn=retrieve_text_from_store)
            ],
            llm=self.llm
        )
        # Create and run the workflow

        workflow = AgentWorkflow(
            agents=[retrieval_agent], root_agent="retrieval agent", timeout=300
        )

        await workflow.run(user_msg="images/image-2.png")


if __name__ == "__main__":
    import asyncio

    adv_mm_rag = MultilingualMultimodalRAG()

    # first time to ingest the data
    # asyncio.run(adv_mm_rag.invoke_ingest_workflow())

    # keep it enabled if you want to ask queries - text2img
    # asyncio.run(adv_mm_rag.invoke_text2img_rag_workflow())

    # keep it enabled if you want to ask queries - img2text
    asyncio.run(adv_mm_rag.invoke_img2text_rag_workflow())
