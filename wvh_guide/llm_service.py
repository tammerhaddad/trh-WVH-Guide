"""
wvh_guide/llm_service.py
------------------------

Provides a wrapper around Google’s Generative AI to convert raw
navigation steps into concise, human-readable directions.
"""

import time
import ast
from typing import List, Tuple

# import google.generativeai as genai
from openai import OpenAI
import os


class Summarizer:
    """
    Wraps a Gemini-based LLM to produce formatted navigation instructions.

    Attributes:
        model (str): The identifier of the Gemini model to use.
    """

    # def __init__(self, api_key: str, model: str) -> None:
    def __init__(self) -> None:
        """
        Configure the Generative AI client and store model choice.

        Args:
            api_key (str): Your Gemini API key.
            model (str): The name or version of the Gemini model
                (e.g. "gemini-2.5-pro-exp-03-25").

        Side Effects:
            Calls `genai.configure()` to authenticate all subsequent requests.
        """
        # genai.configure(api_key=api_key)
        # self.model = model

    def generate(self, steps: List[str]) -> Tuple[List[str], float]:
        """
        Send raw step list to the LLM and parse its formatted response.

        This method:
          1. Builds a prompt that instructs the LLM to produce at most
             seven concise directions in a fixed template.
          2. Calls the Gemini model’s `generate_content` API.
          3. Strips any markdown fences from the reply.
          4. Safely evaluates the result as a Python list of strings.

        Args:
            steps (List[str]): Raw navigation steps (unformatted strings).

        Returns:
            Tuple[List[str], float]:
                - A list of human-readable direction strings.
                - The elapsed time (in seconds) for the LLM call.

        Raises:
            ValueError: If the LLM’s reply cannot be parsed into
                        a `List[str]`, or if it’s malformed.
        """
        start_time = time.time()

        # 1) Build template and inject raw steps via simple replace
        template = """You are embedded in a robot that helps students get from the current location to the goal location.
I have given you some steps to make more human readable. One of the criteria is to make sure the step count is 7 or under.
You have to give the steps clearly and cannot cut short the steps.

The format for each direction (other than elevator) should be: "From {the point}, turn {direction} and go straight to {next point}".
If using an elevator, you must skip over the front of elevator steps and go straight to using the elevator:
For example, instead of saying "From the front of the elevator, turn right and go straight to the elevator."
"Take the elevator to floor 2." "Take a left and go straight to f2_p31.", you should instead say:
"Take the elevator to floor {number}. Take a {direction} and go straight to {next point}" when exiting.
If in front of the destination room, you do not need to say 'From the front of the room, turn right and go straight to the room'.
Same for elevators/stairs/exits.

Return only the steps as a Python list of strings (array of quoted strings). No explanation or extra formatting.

Directions:
{steps_block}
"""
        block = "\n".join(steps)
        prompt = template.replace("{steps_block}", block)

        # 2) Call the LLM
        # model_client = genai.GenerativeModel(self.model)
        # response = model_client.generate_content(prompt)
        # text = response.text.strip()

        # chatGPT version:
        text = self.generate_simple_text(prompt)
        text = text.strip()

        # 3) Remove markdown fences if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])

        # 4) Parse the output as a Python list of strings
        try:
            parsed = ast.literal_eval(text)
            if (
                isinstance(parsed, list)
                and all(isinstance(item, str) for item in parsed)
            ):
                elapsed = time.time() - start_time
                return parsed, elapsed
            else:
                raise ValueError("LLM output is not a List[str]")
        except Exception as err:
            raise ValueError(
                f"Failed to parse LLM response: {err}\nRaw response was:\n{text}"
            )
        

    def generate_simple_text(self, text):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model='gpt-4o-mini',
        )
        res = chat_completion.choices[0].message.content
        return res
