from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import requests
import json
from openai import OpenAI
import httpx
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import Tool
from pydantic import BaseModel, ValidationError
from langchain.tools import tool
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image
import re
import json
from typing import Dict, Any
import sqlite3

# Local module imports
from .agent_state import AgentState, Queries
from .constants import (
    PLAN_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
)

class Agent:
    def __init__(self):
        httpx_client = httpx.Client(http2=True, verify=False, timeout=10.0)
        vcapservices = os.getenv('VCAP_SERVICES')
        if vcapservices:
            try:
                services = json.loads(vcapservices)
                chat_services = list(filter(self.is_chatservice, services.get("genai", [])))
                if chat_services:
                    chat_credentials = chat_services[0]["credentials"]
                    self.model = ChatOpenAI(
                        temperature=0.9,
                        model=chat_credentials["model_name"],
                        base_url=chat_credentials["api_base"],
                        api_key=chat_credentials["api_key"],
                        http_client=httpx_client,
                    )
                    print("VCAP_SERVICES loaded successfully.")
                else:
                    print("No matching chat service found in VCAP_SERVICES")
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from VCAP_SERVICES: {e}")
        else:
            print("VCAP_SERVICES environment variable not found or is empty.")
            # Handle the case where VCAP_SERVICES is not available
            self.model = ChatOpenAI(temperature=0.9, model=os.getenv("OPENAI_API_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), http_client=httpx_client)
        # prompts
        self.PLAN_PROMPT = PLAN_PROMPT
        self.WRITER_PROMPT = WRITER_PROMPT
        self.RESEARCH_PLAN_PROMPT = RESEARCH_PLAN_PROMPT
        self.REFLECTION_PROMPT = REFLECTION_PROMPT
        self.RESEARCH_CRITIQUE_PROMPT = RESEARCH_CRITIQUE_PROMPT
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        # define grpah
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['planner', 'generate', 'reflect', 'research_plan', 'research_critique']
        )

    def extract_json(text):
        # Remove unwanted tags like <think> and <speak>
        cleaned_text = re.sub(r'<\/?[\w\d]+>', '', text).strip()
    
        # Now try to extract the JSON part using regex
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        return json.loads(match.group(0))

    def normalize_to_queries(output: str) -> Dict[str, Any]:
        """
        Normalize LLM output into a dict matching the Queries schema.
        Always returns: {"queries": [...]}.
        Also logs the result as clean JSON.
        """
    
        # Remove <think>...</think> blocks if present
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    
        data: Dict[str, Any]
    
        # Try strict JSON parse first
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and "queries" in parsed:
                data = parsed
            elif isinstance(parsed, list):
                data = {"queries": parsed}
            else:
                raise ValueError("Invalid schema")
        except Exception:
            # Fallback: treat as markdown/bullet/numbered list
            lines = [
                re.sub(r'^\s*[\d\-\*\.\)]*\s*', '', line).strip(' *"`')
                for line in output.splitlines() if line.strip()
            ]
            # Deduplicate while preserving order
            seen = set()
            unique_lines = [q for q in lines if not (q in seen or seen.add(q))]
            data = {"queries": unique_lines}
    
        return data
    def is_chatservice(self, service):
        return service["name"] == "gen-ai-qwen3-ultra"
    
    #implement nodes
    def plan_node(self, state: AgentState):
            messages = [
                SystemMessage(content=self.PLAN_PROMPT),
                HumanMessage(content=state["task"])
            ]
            response = self.model.invoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            # Remove <think>...</think> blocks completely
            response_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
            return {"plan": response_content,
                    "lnode": "planner",
                    "count": 1,
            }
        
    def research_plan_node(self,state: dict):
        """
        Generates a research plan using a Qwen model and Tavily search.
        Works without Pydantic.
        """
        # Invoke Qwen model (plain text output)
        raw_response = self.model.invoke(
            [
                SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
        response_content = getattr(raw_response, "content", str(raw_response))
    
        # Remove <think>...</think> if present
        response_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    
        # Normalize into a list of queries (handle bullets, numbers, etc.)
        lines = [
            re.sub(r'^\s*[\d\-\*\.\)]*\s*', '', line).strip(' *"`')
            for line in response_content.splitlines()
            if line.strip()
        ]
        # Deduplicate
        seen = set()
        queries_list = [q for q in lines if not (q in seen or seen.add(q))]
    
        # Initialize content
        content = state.get("content", [])
    
        # Perform Tavily searches
        for q in queries_list:
            try:
                response = self.tavily.search(query=q, max_results=2)
                for r in response.get("results", []):
                    content_piece = r.get("content", "")
                    if content_piece:
                        content.append(str(content_piece))
            except Exception as e:
                print(f"Search failed for query '{q}': {e}")
    
        return {
            "content": content,
            "queries": queries_list,
            "lnode": "research_plan",
            "count": 1,
        }
    def generation_node(self, state: AgentState):
            content = "\n\n".join(["content"] or [])
            user_message = HumanMessage(content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
            messages = [
                SystemMessage(content=self.WRITER_PROMPT.format(content=content)),
                user_message,
            ]
            response = self.model.invoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            # Remove <think>...</think> blocks completely
            response_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
        
            return {
                "draft": response_content,
                "revision_number": state.get("revision_number", 1) + 1,
                "lnode": "generate",
                "count": 1,
            }
    def reflection_node(self,state: AgentState):
            messages = [
                SystemMessage(content=self.REFLECTION_PROMPT),
                HumanMessage(content=state['draft']),
            ]
            response = self.model.invoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)

            # Remove <think>...</think> blocks completely
            response_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
            return {"critique": response_content,
                    "lnode": "reflect",
                    "count": 1,}
    
    def research_critique_node(self,state: AgentState):
        """
        Generates a research plan using a Qwen model and Tavily search.
        Works without Pydantic.
        """
        # Invoke Qwen model (plain text output)
        raw_response = self.model.invoke(
            [
                SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
                HumanMessage(content=state["critique"]),
            ]
        )
        response_content = getattr(raw_response, "content", str(raw_response))
    
        # Remove <think>...</think> if present
        response_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    
        # Normalize into a list of queries (handle bullets, numbers, etc.)
        lines = [
            re.sub(r'^\s*[\d\-\*\.\)]*\s*', '', line).strip(' *"`')
            for line in response_content.splitlines()
            if line.strip()
        ]
        # Deduplicate
        seen = set()
        queries_list = [q for q in lines if not (q in seen or seen.add(q))]
    
        # Initialize content
        content = state.get("content", [])
    
        # Perform Tavily searches
        for q in queries_list:
            try:
                response = self.tavily.search(query=q, max_results=2)
                for r in response.get("results", []):
                    content_piece = r.get("content", "")
                    if content_piece:
                        content.append(str(content_piece))
            except Exception as e:
                print(f"Search failed for query '{q}': {e}")
    
        return {
            "content": content,
            "queries": queries_list,
            "lnode": "research_critique",
            "count": 1,
        }
    
    def should_continue(self,state):
            if state["revision_number"] > state["max_revisions"]:
                return END
            return "reflect"
