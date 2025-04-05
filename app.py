import os
import re
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# Set your Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client
groq_llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=512,
)

search_tool = DuckDuckGoSearchRun()

def needs_web_search(prompt):
    current_info_keywords = ['latest', 'recent', 'new', 'update', 'current', 'today', 
                             'this month', 'this year', 'changes', 'amendment']
    legal_domains = {
        'tax': ['tax', 'gst', 'income tax', 'taxation'],
        'corporate': ['company', 'corporate', 'business', 'sebi', 'mca'],
        'privacy': ['privacy', 'data protection', 'personal data', 'it act'],
        'environmental': ['environment', 'pollution', 'green', 'climate'],
        'labor': ['labor', 'labour', 'employment', 'worker', 'industrial'],
        'financial': ['finance', 'banking', 'insurance', 'rbi', 'sebi'],
        'property': ['property', 'real estate', 'land', 'housing']
    }
    india_focus = bool(re.search(r'\bindia\b|\bindian\b', prompt.lower()))
    needs_current = any(k in prompt.lower() for k in current_info_keywords)
    legal_terminology = ['law', 'legal', 'regulation', 'policy', 'compliance', 'court', 
                         'judgment', 'notification', 'act', 'rule', 'section', 
                         'legislation', 'statutory']
    has_legal_terms = any(term in prompt.lower() for term in legal_terminology)
    detected_domain = 'general'
    for domain, keywords in legal_domains.items():
        if any(k in prompt.lower() for k in keywords):
            detected_domain = domain
            break
    return (needs_current and has_legal_terms, detected_domain, india_focus)

def search_indian_legal_updates(domain, prompt):
    key_terms = re.sub(r'(latest|recent|new|update|what|tell me about|information on)', '', prompt.lower()).strip()
    search_query = f"latest {domain} law policy legal updates India {key_terms}"
    search_results = search_tool.run(f"{search_query}, num_results=10")
    system_prompt = f"""You are a legal assistant that specializes in Indian law.
Analyze the following search results about {domain} law/policy updates in India related to: {key_terms}

Search query: {search_query}

Search results:
{search_results}

Original user question: {prompt}

Provide a clear, comprehensive response addressing the user's question based on these search results.
Focus on recent legal developments, changes, or updates in India. Structure your response clearly 
with relevant headings and bullet points where appropriate. Include any important dates, authorities 
involved, and stakeholders affected. Note that this information comes from web searches and 
should be verified with official sources.
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = groq_llm.invoke(messages)
    return response.content, search_results

def general_web_search(prompt):
    search_results = search_tool.run(f"{prompt}, num_results=10")
    system_prompt = f"""You are a helpful assistant with access to current information from web searches.
Analyze the following search results related to: {prompt}

Search results:
{search_results}

Original user question: {prompt}

Provide a clear, comprehensive response addressing the user's question based on these search results.
Structure your response clearly with relevant headings and bullet points where appropriate.
Include citations or references to your sources where possible.
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = groq_llm.invoke(messages)
    return response.content, search_results

def direct_llm_response(prompt):
    system_prompt = """You are a knowledgeable assistant.
Provide informative responses based on your training knowledge.
When topics require the most current information, acknowledge that you're limited to 
your training data and recommend the user seek the most recent updates from official sources.
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = groq_llm.invoke(messages)
    return response.content

def process_user_prompt(prompt, force_web_search=False):
    needs_search, domain, india_focus = needs_web_search(prompt)
    if force_web_search or needs_search:
        if india_focus and needs_search:
            response, search_results = search_indian_legal_updates(domain, prompt)
            return response, search_results
        else:
            response, search_results = general_web_search(prompt)
            return response, search_results
    else:
        response = direct_llm_response(prompt)
        return response, None

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Hello, Flask! Use POST /getllm for LLM responses."}

@app.route("/getllm", methods=["POST"])
def getllm():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    user_prompt = data.get("input_string", "")
    web_search = data.get("web_search", False)
    if not user_prompt:
        return jsonify({"error": "No input string provided"}), 400
    response, search_results = process_user_prompt(user_prompt, force_web_search=web_search)
    if search_results:
        return jsonify({
            "response": response,
            "search_performed": True,
            "search_results": search_results
        })
    else:
        return jsonify({
            "response": response,
            "search_performed": False
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
