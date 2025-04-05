# Import necessary libraries
import os
import re
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from pyngrok import ngrok

# Set your Groq API key
# Replace with your actual API key or use environment variables
GROQ_API_KEY = "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"  
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the Groq client with a supported model
groq_llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=512,
)

# Set up DuckDuckGo search
search_tool = DuckDuckGoSearchRun()

def needs_web_search(prompt):
    """
    Determine if the prompt requires a web search for latest information
    
    Args:
        prompt: User's prompt/question
        
    Returns:
        Boolean indicating if web search is needed and detected domain
    """
    # Keywords that suggest need for current information
    current_info_keywords = ['latest', 'recent', 'new', 'update', 'current', 'today', 
                            'this month', 'this year', 'changes', 'amendment']
    
    # Legal domains to detect
    legal_domains = {
        'tax': ['tax', 'gst', 'income tax', 'taxation'],
        'corporate': ['company', 'corporate', 'business', 'sebi', 'mca'],
        'privacy': ['privacy', 'data protection', 'personal data', 'it act'],
        'environmental': ['environment', 'pollution', 'green', 'climate'],
        'labor': ['labor', 'labour', 'employment', 'worker', 'industrial'],
        'financial': ['finance', 'banking', 'insurance', 'rbi', 'sebi'],
        'property': ['property', 'real estate', 'land', 'housing']
    }
    
    # Check if prompt has India focus
    india_focus = bool(re.search(r'\bindia\b|\bindian\b', prompt.lower()))
    
    # Check if prompt suggests need for current information
    needs_current = any(keyword in prompt.lower() for keyword in current_info_keywords)
    
    # Look for legal terminology
    legal_terminology = ['law', 'legal', 'regulation', 'policy', 'compliance', 'court', 
                         'judgment', 'notification', 'act', 'rule', 'section', 
                         'legislation', 'statutory']
    has_legal_terms = any(term in prompt.lower() for term in legal_terminology)
    
    # Detect domain
    detected_domain = 'general'
    for domain, keywords in legal_domains.items():
        if any(keyword in prompt.lower() for keyword in keywords):
            detected_domain = domain
            break
    
    # Determine if web search needed
    return (needs_current and has_legal_terms, detected_domain, india_focus)

def search_indian_legal_updates(domain, prompt):
    """
    Search for Indian legal updates based on domain and prompt
    
    Args:
        domain: Detected legal domain
        prompt: Original user prompt
        
    Returns:
        Analysis of legal updates and search results
    """
    # Extract key terms from the prompt to enhance search
    key_terms = re.sub(r'(latest|recent|new|update|what|tell me about|information on)', '', prompt.lower())
    key_terms = key_terms.strip()
    
    # Construct search query
    search_query = f"latest {domain} law policy legal updates India {key_terms}"
    
    # Perform the search with at least 10 results
    search_results = search_tool.run(f"{search_query}, num_results=10")
    
    # Create prompt for legal updates
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
    
    # Send to LLM for analysis
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = groq_llm.invoke(messages)
    return response.content, search_results

def general_web_search(prompt):
    """
    Perform a general web search for any topic
    
    Args:
        prompt: User's prompt/question
        
    Returns:
        Analysis of search results and the search results themselves
    """
    # Perform the search with at least 10 results
    search_results = search_tool.run(f"{prompt}, num_results=10")
    
    # Create prompt for analysis
    system_prompt = f"""You are a helpful assistant with access to current information from web searches.
    Analyze the following search results related to: {prompt}
    
    Search results:
    {search_results}
    
    Original user question: {prompt}
    
    Provide a clear, comprehensive response addressing the user's question based on these search results.
    Structure your response clearly with relevant headings and bullet points where appropriate.
    Include citations or references to your sources where possible.
    """
    
    # Send to LLM for analysis
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = groq_llm.invoke(messages)
    return response.content, search_results

def direct_llm_response(prompt):
    """
    Get a direct response from the LLM without web search
    
    Args:
        prompt: User's prompt/question
        
    Returns:
        LLM response
    """
    system_prompt = """You are a knowledgeable assistant.
    Provide informative responses based on your training knowledge.
    When topics require the most current information, acknowledge that you're limited to 
    your training data and recommend the user seek the most recent updates from official sources.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = groq_llm.invoke(messages)
    return response.content

def process_user_prompt(prompt, force_web_search=False):
    """
    Main function to process user prompts - decides whether to use web search or direct response
    
    Args:
        prompt: User's prompt/question
        force_web_search: Boolean to force web search regardless of prompt analysis
        
    Returns:
        Response to the user's prompt and search results if applicable
    """
    # Determine if web search is needed for Indian legal content
    needs_search, domain, india_focus = needs_web_search(prompt)
    
    if force_web_search or needs_search:
        if india_focus and needs_search:
            # For Indian legal topics
            response, search_results = search_indian_legal_updates(domain, prompt)
            return response, search_results
        else:
            # For general web search
            response, search_results = general_web_search(prompt)
            return response, search_results
    else:
        # Direct LLM response without search
        response = direct_llm_response(prompt)
        return response, None

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Hello, Flask! Use POST /getllm for LLM responses."}

@app.route("/getllm", methods=["POST"])
def getllm():
    data = request.get_json()  # Parse the incoming JSON request
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_prompt = data.get("input_string", "")  # Get the string from the JSON
    web_search = data.get("web_search", False)  # Get the web_search flag, default to False
    
    if not user_prompt:
        return jsonify({"error": "No input string provided"}), 400
    
    response, search_results = process_user_prompt(user_prompt, force_web_search=web_search)
    
    # Return the response and search results if available
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

# Run this cell to start the Flask server with ngrok
# This will create a public URL that can be accessed from anywhere
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Run the Flask app
# Note: This uses the 'run_simple' server from Werkzeug which is suitable for development
from werkzeug.serving import run_simple
run_simple('localhost', 5000, app, use_reloader=False, use_debugger=False)