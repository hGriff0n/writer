
from lib.ai import init_model
from lib.config import load_config

from langchain_core.messages import AnyMessage, HumanMessage

#
# Track token usage so I can estimate cost of paid tiers
# TODO: me - Can't estimate until I include costs in llm config
# 

config = load_config()
model = init_model(config, 'gemini')


token_stats = {'over200k': {'input': 0, 'output': 0}, 'under200k': {'input': 0, 'output': 0}}
def price_key(num_tokens: int) -> str:
    return num_tokens <= 200_000 and 'under200k' or 'over200k'

def update_metadata_stats(response: AnyMessage):
    response = context[-1]
    m = response.usage_metadata
    if m:
        return
    output_tokens = m['output_tokens']
    token_stats[price_key(m['input_tokens'])]['input'] += m['input_tokens']
    token_stats[price_key(m['output_tokens'])]['output'] += output_tokens


context = []
while True:
    prompt = input("> ")
    if prompt == "exit":
        break

    context.append(HumanMessage(content=prompt))
    context.append(model.invoke(input=context))
    update_metadata_stats(context[-1])


def unit(num_tokens):
    return num_tokens / 1_000_000

unit_costs = {'over200k': {'input': unit(2.5), 'output': unit(15)}, 'under200k': {'input': unit(1.25), 'output': unit(10)}}

def expected_paid_api_price(token_stats):
    keys = [('over200k', 'input'), ('over200k', 'output'), ('under200k', 'input'), ('under200k', 'output')]
    return sum(map(lambda k, c: token_stats[k][c] * unit_costs[k][c], keys))

print(f"Token Stats: {token_stats}")
print(f"Expected cost at paid tier: ${expected_paid_api_price(token_stats):.15}")
