from openai import OpenAI
import time

client = OpenAI(api_key="sk-Zzu4LQYbNTNsDoyhaVKoUvtuk8oEruffE6CjhmM2fL6vOzar", 
                base_url="http://oneapi-bcloud.bc-inner.com/v1")
# response = client.chat.completions.create(
#     # model='Pro/deepseek-ai/DeepSeek-R1',
#     model="deepseek-r1",
#     messages=[
#         {'role': 'user', 
#         'content': "The twelve letters $A$,$B$,$C$,$D$,$E$,$F$,$G$,$H$,$I$,$J$,$K$, and $L$ are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and then those six words are listed alphabetically. For example, a possible result is $AB$, $CJ$, $DG$, $EK$, $FL$, $HI$. The probability that the last word listed contains $G$ is $\\frac mn$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$. Let's think step by step and output the final answer within \\boxed{}."}
#     ],
#     stream=True,
#     timeout=600
# )

# start = time.time()

# for chunk in response:
#     if not chunk.choices:
#         continue
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)

# time_taken1 = time.time() - start

start = time.time()

response = client.chat.completions.create(
    # model='Pro/deepseek-ai/DeepSeek-R1',
    model="deepseek-r1",
    messages=[
        {'role': 'user', 
        'content': "The twelve letters $A$,$B$,$C$,$D$,$E$,$F$,$G$,$H$,$I$,$J$,$K$, and $L$ are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and then those six words are listed alphabetically. For example, a possible result is $AB$, $CJ$, $DG$, $EK$, $FL$, $HI$. The probability that the last word listed contains $G$ is $\\frac mn$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$. Let's think step by step and output the final answer within \\boxed{}."}
    ],
    stream=False,
    timeout=60000
)

time_taken2 = time.time() - start

# print(time_taken1, time_taken2)

print(time_taken2)