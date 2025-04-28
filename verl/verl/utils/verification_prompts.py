
VERIFICATION_PROMPT_1 = """You are an excellent teacher and good at judgeing whether your student's answer is right or not when providing a problem. Now given a problem and its solution, verify the correctness of solution step by step. {think_hint}At the end of the solution verification, write it in the from \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.
{question_pair}{think_pair}{solution_pair}
"""

VERIFICATION_PROMPT_2 = """Given a problem and its solution, verifying correctness step by step. {think_hint}At the end of the solution verification, write it in the form \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.
{question_pair}{think_pair}{solution_pair}
"""

THINK_HINT = """Meanwhile, there is a thinking progress contains contemplation and self-reflection when trying to solve the problem. You can dive into the thinking progress and obtain key information from it which is helpful to judge whether the solution is correct. """


prompt_dict = {
    "verification_v1": VERIFICATION_PROMPT_1,
    "verification_v2": VERIFICATION_PROMPT_2
}