import ollama

documents = [
    "Server Alpha login policy: Always use SSH keys. Passwords are disabled.",
    "The admin contact for the backup cluster is Jadi at admin@local.mesh.",
    "Project Orion deployment target date is set for October 10th.",
    "The office cafeteria serves lunch from 12:00 to 14:00 on weekdays.",
    "Parking permits must be renewed every January at the front desk.",
    " The quarterly all-hands meeting is held in the main auditorium.",
    "Employees get 20 days of paid vacation per year, plus public holidays.",
    "The recommended font for internal presentations is Inter, size 14.",
    "Printer on the third floor is out of toner; use the second floor one.",
    " Team building event: bowling night on the last Friday of the month."
    "The company book club is currently reading 'The Phoenix Project'.",
    "Coffee machine maintenance is scheduled every Wednesday morning.",
    "The gym in the basement is open from 6:00 to 22:00 for all staff.",
]

user_query = "what is the name and email address of the person responsible for the back up cluster"
system_prompt = (
    "You are the helpdesk assistant of our company, answer briefly and directly, no disclaimer. Use only this context:"
    + "\n".join(documents)
)
# print(system_prompt)
# exit()
# print(ollama.list())

model_name = "qwen3:30b-a3b-instruct-2507-q4_K_M"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
]
response = ollama.chat(model=model_name, messages=messages)
print(response["message"]["content"])

# Token usage, as counted by the Ollama server itself.
prompt_tokens = response["prompt_eval_count"]
completion_tokens = response["eval_count"]
model_info = ollama.show(model_name).modelinfo
context_length = next(v for k, v in model_info.items() if k.endswith(".context_length"))

print(f"\nprompt (sent):      {prompt_tokens} tokens")
print(f"response (received): {completion_tokens} tokens")
print(f"total:               {prompt_tokens + completion_tokens} tokens")
print(
    f"context window:      {context_length} tokens "
    f"({(prompt_tokens + completion_tokens) / context_length:.2%} used)"
)
