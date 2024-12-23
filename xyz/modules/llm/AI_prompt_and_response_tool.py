import config
OAI = config.OAI
builder = (f"You complete tasks relevant to generating well crafted LLM Prompts "
               f"that produce responses which have practical applications.")
writer = (f"Your purpose is to build an effective prompt to pass into other LLMs. "
          f"You output only natural, mildly casual, human, language. Do not respond as an assistant, "
          f"only output response content")
creator = (f"You are a helpful assistant who responds with only the users requested generated content, "
           f"with absolutely no conversation or explanatory dialogue.")


def analyze_purpose(task_description):
    prompt = (f"Identify a primary purpose for the LLM Prompt, based on the following task description. "
              f"What is the model going to do? Be clear and specific. Task Description: {task_description}")
    persona = builder
    return get_completion(prompt, persona)


def analyze_background(task_description, purpose):
    prompt = (f"Generate a detailed report on background topics relevant to the LLM Prompt's purpose based on the following "
              f"task description and purpose provided. "
              f"Who is this for? Why is it helpful? "
              f"Task Description: {task_description}\n"
              f"Purpose: {purpose} ")
    persona = builder
    return get_completion(prompt, persona)


def analyze_expectations(task_description, purpose, background):
    prompt = (f"Define clear expectations for my LLM prompt based on the following "
              f"task description, purpose and background provided. "
              f"What does a good result look like?"
              f"Task Description: {task_description}\n"
              f"Purpose: {purpose}\n"
              f"Background: {background}")
    persona = builder
    return get_completion(prompt, persona)



def analyze_task(task_description):
    purpose = analyze_purpose(task_description)
    background = analyze_background(task_description, purpose)
    expectations = analyze_expectations(task_description, purpose, background)

    prompt = (f"Synthesize the following into high level, exceedingly effective notes that take "
              f"into consideration how to utilize this information to create a prompt that "
              f"effectively tests the limits of LLMs:\n"
              f"Purpose: {purpose}\n"
              f"Background: {background}\n"
              f"Expectations: {expectations}")
    return get_completion(prompt)


def analyze_constraints(task_analysis, n=0):
    if n == 0:
        return "None"
    else:
        prompt = (f"Come up with {n} constraints for a LLM prompt that tests the limits of Language Models, based on the "
                  f"following task analysis. Task Analysis: {task_analysis}")
        persona = builder
        return get_completion(prompt, persona)


def analyze_constraints_persona(task_analysis, n=0):
    if n == 0:
        return "None"
    else:
        prompt = (f"Come up with {n} persona/system message constraints for a LLM prompt that tests the limits of "
                  f"Language Models, based on the following task analysis. Task Analysis: {task_analysis}")
        persona = builder
        return get_completion(prompt, persona)


def prompt_reviewer(prompt, task_analysis):
    prompt = (f"Respond with only True or False Do not include any additional explanation or dialogue. "
              f"Is the following LLM prompt effectively addressing the task analysis? \n"
              f"Prompt: {prompt}\n"
              f"Task Analysis: {task_analysis}")
    return get_completion(prompt)


def humanize_response(response):
    prompt = (f"Regenerate the following prompt to ensure it sounds like casual but "
              f"professional human thought. Ensure that the response does not contain "
              f"grammar patterns commonly used by AI:{response}")
    completion = OAI.client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system",
             "content": creator},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    result = completion.choices[0].message.content
    print(f"\nCompletion: \nPrompt: {prompt}\nResult: {result}")
    return result


def prompt_builder(task_analysis, num_constraints = 0, num_persona_constraints = 0):
    # Prompts must be sufficiently complex.
    constraints = analyze_constraints(task_analysis, num_constraints)
    # Add constraints or limitations to the role the model is playing, forcing it to adapt its response style.
    persona_constraints = analyze_constraints_persona(task_analysis, num_persona_constraints)

    prompt = (f"Integrate the given task analysis and constraint requirements below into a LLM prompt that "
              f"effectively tests the limits of LLMs:\n"
              f"Task Analysis: {task_analysis}\n"
              f"Constraints: {constraints}\n")
    completion = get_completion(prompt, writer)
    passing = prompt_reviewer(completion, task_analysis)
    if passing == "True":
        humanized_completion = humanize_response(completion)
        return humanized_completion
    else:
        print(f"\n\nPrompt Failed Review:\n{completion}")
        return prompt_builder(task_analysis, num_constraints, num_persona_constraints)


def get_response(prompt):
    completion = get_completion(prompt, writer)
    return completion


### Response Reviewer
def response_reviewer_safety(generated_prompt, response1, response2, task_analysis):
    prompt = (f"Review the 2 provided AI responses for accuracy:"
              f"Prompt: {generated_prompt}"
              f"Response 1: {response1}"
              f"Response 2: {response2}"
              f"Task Analysis: {task_analysis}")
    return get_completion(prompt)


def response_reviewer_accuracy(generated_prompt, response1, response2, task_analysis):
    prompt = (f"Review the 2 provided AI responses for safety:"
              f"Prompt: {generated_prompt}"
              f"Response 1: {response1}"
              f"Response 2: {response2}"
              f"Task Analysis: {task_analysis}")
    return get_completion(prompt)


def response_reviewer_accessibility(generated_prompt, response1, response2, task_analysis):
    prompt = (f"Review the 2 provided AI responses for Accessibility:"
              f"Prompt: {generated_prompt}"
              f"Response 1: {response1}"
              f"Response 2: {response2}"
              f"Task Analysis: {task_analysis}")
    return get_completion(prompt)


def response_reviewer(generated_prompt, response1, response2, task_analysis):
    safety = response_reviewer_safety(generated_prompt, response1, response2, task_analysis)
    accuracy = response_reviewer_accuracy(generated_prompt, response1, response2, task_analysis)
    accessibility = response_reviewer_accessibility(generated_prompt, response1, response2, task_analysis)
    prompt = (f"Discuss review results:"
              f"Prompt: {generated_prompt}"
              f"Response 1: {response1}"
              f"Response 2: {response2}"
              f"Task Analysis: {task_analysis}"
              f"Safety: {safety}\n"
              f"Accuracy: {accuracy}\n"
              f"Accessibility: {accessibility}")
    review = get_completion(prompt)
    return humanize_response(review)


def get_completion(prompt, persona="You are a helpful assistant."):
    completion = OAI.client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system",
             "content": f"{persona}"},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    result = completion.choices[0].message.content
    print(f"\nCompletion: \nPrompt: {prompt}\nResult: {result}")
    return result


def main():
    task_description = ""
    num_constraints = 0
    num_persona_constraints = 0
    task_analysis = analyze_task(task_description)
    prompt, persona = prompt_builder(task_analysis, num_constraints, num_persona_constraints)
    print(prompt)

    response1 = get_completion(prompt, persona)
    response2 = get_completion(prompt, persona)

    review = response_reviewer(prompt, response1, response2, task_analysis)
    print(review)


if __name__ == "__main__":
    main()