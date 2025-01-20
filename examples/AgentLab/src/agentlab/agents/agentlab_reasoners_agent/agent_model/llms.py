import openai
from typing import Callable, Tuple, Optional


def IDENTITY(x):
    return x, True, None


class LLM:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = openai.Client(api_key=api_key)

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_image: str = None,
        parser: Callable[[str], Tuple[str, bool, Optional[str]]] = IDENTITY,
        **kwargs,
    ):

        print("KWARSG BEING PASSED TO OPENAI API")
        # FIEXME: hardcode
        kwargs["temperature"] = 0.7
        print(kwargs)
        response = None
        if base64_image is None:
            print("1 - using text only")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
        else:
            print("2 - screenshot passed in as well")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    },
                ],
                **kwargs,
            )

        answer_dicts = []
        for choice in response.choices:
            content = choice.message.content
            parsed_content = parser(content)
            answer_dict = parsed_content[0]
            answer_dicts.append(answer_dict)

        return answer_dicts
