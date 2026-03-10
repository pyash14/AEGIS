import anthropic
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_claude(system_prompt: str, user_message: str, history: list = None) -> str:
    '''
    Call Claude API. Returns string response.
    history: list of {role, content} dicts for multi-turn
    '''
    try:
        messages = []
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': user_message})

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    except anthropic.APIConnectionError as e:
        return f'Connection error: {str(e)}'
    except anthropic.RateLimitError:
        return 'Rate limit reached. Please wait a moment.'
    except Exception as e:
        return f'Unexpected error: {str(e)}'


if __name__ == '__main__':
    response = call_claude(
        system_prompt='You are a helpful assistant.',
        user_message='Say hello in one sentence.'
    )
    print('Claude response:', response)