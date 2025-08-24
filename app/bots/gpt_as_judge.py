from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from models.prompt import PromptManager
from utils.logs import set_logger


logger = set_logger(__name__)



class JudgeResponse(BaseModel):
    label: str = Field(..., description="The label of the answer, e.g., A, B, C, D")
    reason: str = Field(..., description="The reason for the judgment")



def check_label(query: str, answer: str) -> str:
    """
    Check the label of the answer based on the provided query and answer text.

    Args:
        query (str): The question text containing multiple-choice options.
        answer (str): The assistant's answer text.

    Returns:
        str: The determined label (A, B, C, D) or "Unknown" if no match is found.
    """
    assistant = init_chat_model("openai:gpt-4.1").with_structured_output(JudgeResponse)
    prompt_temp = PromptManager("check-label", locale="en")
    prompt = ChatPromptTemplate([
        SystemMessage(content=prompt_temp.system),
        HumanMessage(content=prompt_temp.human)  # .format(query="{query}", answer="{answer}"))
    ])
    chain = prompt | assistant

    logger.debug(f"{prompt = }")
    logger.debug(f"{query = }")
    logger.debug(f"{answer = }")
    
    response = chain.invoke({"query": query, "answer": answer})
    logger.debug(f"Judge Response: {response}")
    return response


__all__ = ["check_label"]