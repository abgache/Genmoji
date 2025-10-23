if __name__ == "__main__":
    from llm import *
    from time_log import time_log_module as tlm
else:
    from scripts.llm import *
    from scripts.time_log import time_log_module as tlm

def enhance_prompt(emoji_description, v=True, local=True):
    # Real genmoji prompt (got on a mac)
    prompt = f"""You are helping create a prompt for a Emoji generation image model. An emoji must be easily interpreted when small so details must be exaggerated to be clear. Your goal is to use descriptions to achieve this.

    You will receive a user description, and you must rephrase it to consist of short phrases separated by periods, adding detail to everything the user provides, Write long prompts. 

    Add describe the color of all parts or components of the emoji. Unless otherwise specified by the user, do not describe people. Do not describe the background of the image. Your output should be in the format:

    emoji of *description*. *addon phrases*. 3D lighting. no cast shadows.
    The description should be a 1 sentence of your interpretation of the emoji. Then, you may choose to add addon phrases. You must use the following in the given scenarios:

    "cute.": If generating anything that's not an object, and also not a human
    "enlarged head in cartoon style.": ONLY animals
    "head is turned towards viewer.": ONLY humans and animals
    "detailed texture.": ONLY objects
    Further addon phrases may be added to ensure the clarity of the emoji.
    In your answer, no markdown should be used, and generate ONLY the emoji description 
    There is the emoji description : {emoji_description}
    Output only the emoji prompt: """
    if not local:
        if v:
            print(f"{tlm()} Enhancing prompt using Gemini 2.0 Flash...")
        ak = None
        if ak == None:
            raise ValueError(f"{tlm()} You can't use the Gemini 2.0 Flash API whitout an api key.")
        prompt = gemini_ai(prompt, api_keya=ak)
    elif local:
        if v:
            print(f"{tlm()} Enhancing prompt using LLaMa3.1:8b")
        prompt = local_ai(prompt)
    return prompt

if __name__ == "__main__":
    print(f'Prompt enhancer test:\n"A flying pig" becomes {enhance_prompt("A flying pig", local=True)}')
