from datasets import load_dataset
import openai
import asyncio
from asyncio import Semaphore
from datasets import Dataset
from huggingface_hub import HfApi

#client1 = openai.OpenAI(api_key=api_key)
client = openai.AsyncOpenAI(api_key=api_key)
#can you do some more filtering to make sure you get better personas 
#and prompts?

def preProcessPersonas(seed, n):
    dataset = load_dataset("proj-persona/PersonaHub","persona")["train"]

    def filter_by_completion_length(x, min_length, max_length):
        completion = x["persona"]
        length = len(completion)
        return min_length <= length <= max_length
    
    filtered_dataset = dataset.filter(
    lambda x: filter_by_completion_length(x, min_length=85, max_length=300))


    return filtered_dataset.shuffle(seed).select(range(n))["persona"]

def preProcessPrompts(seed, n):
    dataset = load_dataset("abhayesian/ultrachat-first-response")["train_sft"]

    def filter_by_completion_length(x, min_length, max_length):
        completion = x["prompt"]
        length = len(completion)
        return min_length <= length <= max_length
    
    filtered_dataset = dataset.filter(
    lambda x: filter_by_completion_length(x, min_length=62, max_length=300))

    def choose_questions(x):
        if "?" in x["prompt"]:
            return True
        else:
            return False 
    
    question_dataset = filtered_dataset.filter(lambda x: choose_questions(x))

    return question_dataset.shuffle(seed).select(range(n))["prompt"]

def persona_transform(persona):
    query_prompt = f"""Transform the following persona into an abstract personality description with no occupational references in the style of a system prompt:

    Original persona: {persona}

    Provide a 2-4 system prompt that captures their personality traits, behavioral patterns, motivations, and worldview without mentioning any specific job, field, industry, or professional role. Write it in the format of a system prompt for an LLM.

    Provide the new system prompt here:"""
    return query_prompt

def prompt_transform(prompt):
    query_prompt = f"""Transform the following fact-based question into an opinion-based question that would elicit different responses from people with different personalities:

    Original question: {prompt}
    Provide a reformulated question that focuses on personal perspective, preferences, judgments, or values rather than factual knowledge. The question should allow people with different personalities and worldviews to answer in meaningfully different ways.
    Write the new question here:"""
    
    return query_prompt

def persona_system_prompt_transform(persona):
    query_prompt = f"""

You are roleplaying as a specific person with distinct personality traits, values, and worldview. 

Your personality profile:
{persona}

When answering questions:
- Respond authentically from this persona's perspective
- Express genuine opinions, preferences, and judgments this person would hold
- DO NOT say "as an AI" or disclaim that you lack personal experience
- Fully embody this personality's way of thinking and speaking
- Give answers this specific person would give, not generic balanced responses
- Be opinionated and committed to this worldview

Answer the following question as this person would:
"""

    return query_prompt



async def query_llm(prompt, system_prompt=None, model="gpt-4o-mini", temperature=0.7, max_tokens=100):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

async def batch_query_llm(prompts, system_prompt=None, model="gpt-4o-mini", temperature=0.7):
    tasks = [query_llm(prompt, system_prompt, model, temperature, 150) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

async def batch_query_llm_dataset(prompts, system_prompts, model="gpt-4o-mini", temperature=0.7):
    tasks = []
    for prompt in tqdm(prompts):
        for persona in system_prompts:
            tasks.append(query_llm(prompt, persona_system_prompt_transform(persona), model, temperature, 300))
    results = await asyncio.gather(*tasks)

    return results


def gen_prompts():
    print("preparing prompts")
    prompts = preProcessPrompts(42, 100)
    outPrompts = []

    for prompt in prompts:
        outPrompts.append(prompt_transform(prompt))
    print("prompt prepared")

    with open("./scripts/datasetCreate/promptSystemPrompt.txt") as g:
        system_prompt_prompt = g.read()

    print("generating prompts")
    results = asyncio.run(batch_query_llm(outPrompts, system_prompt_prompt))
    print("prompts generated")
    return results

def gen_personas():
    print("preparing personas")
    personas = preProcessPersonas(42, 6)
    outPersonas = []

    for persona in personas:
        outPersonas.append(persona_transform(persona))
    print("personas prepared")

    with open("./scripts/datasetCreate/personaSystemPrompt.txt") as g:
        system_prompt_persona = g.read()

    print("generating personas")
    results = asyncio.run(batch_query_llm(outPersonas, system_prompt_persona))
    
    print("personas generated")
    return results

def gen_dataset():
    prompts = gen_prompts()
    personas = gen_personas()

    print("generating completions")
    results = asyncio.run(batch_query_llm_dataset(prompts, personas))
    print("completions complete")
    idx = 0
    outData = {"persona": [], "prompt":[], "completions":[]}
    for i in personas:
        for j in prompts:
            outData["persona"].append(i)
            outData["prompt"].append(j)
            outData["completions"].append(results[idx])
            idx += 1


    dataset = Dataset.from_dict(outData)
    dataset.push_to_hub("desh2806/emgMisalgGen-xlarge", token=token)
 
gen_dataset()

