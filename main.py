
from flask import Flask, jsonify, request
import requests
import os
from langchain_openai import OpenAI
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# The print_messages function the user provided (added here)
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ JARVIS TOOLS RESULT:  {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\n🤖 JARVIS: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"\n🙋 USER: {message.content}")
            
    for message in reversed(messages):
        if (isinstance(message, AIMessage)):
            return message.content

# @tool
def search_web(query):
    """Searches the web using Google Custom Search JSON API"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query
    }
    response = requests.get(url, params=params).json()
    summarized_results = []
    if response and "items" in response:
        for item in response["items"]:
            summarized_results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link")
            })
        return summarized_results
    else:
        return f"Error Occurred or the response is empty: {response}"

model = ChatOpenAI(model="gpt-4o").bind_tools(tools=[search_web])

def run_ai_agent(state: AgentState):
    system_prompt = SystemMessage(content="""
       You are Graduation Mate, an AI agent exclusively for high school students seeking extracurricular recommendations and college guidance.

YOUR PURPOSE:
Provide detailed, actionable guidance on extracurricular activities, programs, competitions, clubs, and volunteering opportunities that align with a student’s grade level, current activities, and intended college majors.

TOOLS:
- You have access to a tool called 'search_web' which can search the web for official or trustworthy links. Use it to provide real links for each recommendation.

RESPONSE RULES:

1. OUTPUT FORMAT:
   - Return a single HTML fragment wrapped in <div>.
   - Each recommendation is a **card** inside this div.
   - Card must include:
     - Title of the opportunity
     - Category/type (Competition, Club, Volunteering, Program, Online, etc.)
     - 1–3 sentence description
     - Skills gained (displayed as chips/tags)
     - Time commitment or grade level
     - **A clickable header ** displayed below, in blue, using standard <a> tag. Example:
   - Cards should have rounded corners, subtle borders, and consistent dark gradient styling.
   - Use inline CSS only.

2. CONTENT RULES:
   - Provide **5–6 real extracurricular recommendations** per request.
   - Each card is one opportunity.
   - Use the 'search_web' tool to fetch a real link for each recommendation.
   - Tone: encouraging, professional, high school appropriate.

3. BEHAVIOR RULES:
   - Never provide non-extracurricular advice.
   - Only suggest real, actionable opportunities.
   - Do not include scripts, iframes, or unsafe HTML.
   - Only return the HTML fragment (no full HTML page, no <head> or <body>).

4. STYLING GUIDELINES:
   - Background gradient: #1B1B1B → #003153
   - Card: border-radius 20px, subtle border, padding 24px, margin-bottom 20px
   - Text: white (#ffffff)
   - Accent colors for skills or headers: #764FF5 (purple) and #76FBA6 (green)
   - Skill tags: rounded backgrounds with small padding

EXAMPLE CARD STRUCTURE:

<div style="background:linear-gradient(135deg,#1B1B1B,#003153); border-radius:20px; padding:24px; margin-bottom:20px; border:1px solid rgba(255,255,255,0.1);">
  <h2 style="color:#764FF5;"> <a href="https://www.firstinspires.org/robotics/frc" style="color:#1E90FF; text-decoration:underline; text-decoration-color:#1E90FF;">Robotics Club </a></h2>
  <p>Join your school's robotics club to compete in FIRST competitions. Gain programming, teamwork, and engineering experience.</p>
  <div style="display:flex; gap:8px; flex-wrap:wrap;">
    <span style="background:rgba(118,251,166,0.2); color:#76FBA6; padding:4px 8px; border-radius:12px;">Programming, </span>
    <span style="background:rgba(118,251,166,0.2); color:#76FBA6; padding:4px 8px; border-radius:12px;">Engineering, </span>
    <span style="background:rgba(118,251,166,0.2); color:#76FBA6; padding:4px 8px; border-radius:12px;">Teamwork</span>
  </div>
  <p>Grade Level: 9–12 • 5 hrs/week</p>
  
</div>

Remember: **each card must have a real link**. Make it with the heading and keep the underline color the same blue as the heading.

And always remember to be specific in every recommendation. Dont give the user too vague recommendations like Hackathons, instead provide specific hackathons.
`And also put a comma between the set of skills you give like for eg, in the example:  Programming Engineering Teamwork you would display it as Programming, Engineering, Teamwork


AND MOST IMPORTANTLY EVERYTHING U GIVE SHOULD BE THE UI / HTML, NO OTHER RESPONSE BUT DONT PUT LIKE A "'''HTML" IN FRONT OF THE ACUTAL HTML THAT'S NOT NEEDED, JUST THE PURE HTML MAN.

AND AGAIN IM TELLING U DONT PROVIDE ANYTHING EXCEPT FOR THE HTML NO TEXT NOTHING. EVEN IF U WERNET ABLE TO ACCESS REAL TIME DATA, DONT MENTION THAT TO THE USER



    """)
    messages = [system_prompt] + list(state['messages'])
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and isinstance(last_message, AIMessage) and last_message.tool_calls):
        return "invoke_tool"
    elif isinstance(last_message, ToolMessage):
        return "process"
    else:
        return "done"

# Build the graph
graph = StateGraph(state_schema=AgentState)
graph.add_node("search_tool", ToolNode(tools=[search_web]))
graph.add_node("run_agent", run_ai_agent)

graph.add_edge(START, "run_agent")
graph.add_conditional_edges("run_agent", should_continue, {
    "process": "run_agent",
    "done": END,
    "invoke_tool": "search_tool"
})
graph.add_edge("search_tool", "run_agent")

app_graph = graph.compile()

def get_final_ai_message(messages):
    """Extract the final AI message content"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return "No response generated"

server = Flask(__name__)

@server.route("/", methods=["GET", "POST"])
def run_ecrecommender():
    data = request.get_json()
    if data:
        user_response = data.get("user_response")
        if user_response:
            print("\n ===== EC RECOMMENDER =====")
            # Use the compiled graph instead of run_ai_agent directly
            initial_state = AgentState(messages=[HumanMessage(content=user_response)])
            final_state = app_graph.invoke(initial_state)
            
            # Use the print_messages helper to print the last few messages and get the last AI content
            printed_ai = print_messages(final_state["messages"])
            # fallback to get_final_ai_message if print_messages returned None
            final_ai_message = printed_ai or get_final_ai_message(final_state["messages"])
            
            print(f"Final AI Message: {final_ai_message}")
            print("\n ===== EC RECOMMENDER DONE =====")
            return jsonify({"reply": final_ai_message})
    
    # Return the error HTML if no user_response
    html = """[Your HTML error message here]"""
    return jsonify({"reply": html})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    gunicorn main:server --bind 0.0.0.0:$PORT --workers 1

