# pip install ag2 autogen openai
from autogen import ConversableAgent
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

print("Imported successfully")

# Setting configuration
configuration =  {
    "model": "gpt-4o-mini",
    "api_key":  os.getenv('OPENAI_API_KEY')
}

# Configure a Conversable Agent that will never ask for our input and uses the config provided
nba_fan = ConversableAgent(
    name="nba", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    system_message="you are an nba fan",
)

soccer_fan = ConversableAgent(
    name="soccer", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    system_message="you are a soccer fan",

)

# This time,  initiate_chat() function will be used and we can start a chat between them.
# We need to first set the recipient of the message.
# The initial message
# Number of turns after what the conversation will stop.
# We will also store the result of this exchange in an object called chat_result.

result = nba_fan.initiate_chat(
     recipient = soccer_fan,
     message="convince me that soccer is better than nba",
     max_turns=2 # The conversation will stop after each agent has spoken twice
 )
print(result)

#
import pprint
pprint.pprint(result.chat_history)
pprint.pprint(result.cost)
#send method
print("----------------")
nba_fan.send(message="What was the last statement we made?",recipient=soccer_fan)

# Generate the summary of conversion with summary_method parameter
result = nba_fan.initiate_chat(
    recipient = soccer_fan,
    message="convince me that soccer is better than nba",
    max_turns=1, # The conversation will stop after each agent has spoken twice
    summary_method="reflection_with_llm", # Can be "last_message" (DEFAULT) or "reflection_with_llm"
    summary_prompt="Summarize the conversation", # We specify the prompt used to summarize the chat
)

pprint.pprint(result.summary)
