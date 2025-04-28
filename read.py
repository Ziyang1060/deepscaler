# import pandas as pd

# data = pd.read_parquet("/data_train/code/search/zhaoyufan/deepscaler/data/aime.parquet")
# for idx, item in data.iterrows():
#     for k, v in item.items():
#         print('{}: {}'.format(k, v))
#     print(len(item['data_source']))
#     if idx == 1:
#         break   

# import re
# res = """walks at 3 km/h, including the coffee shop time t.\n\nFirst, compute the walking time at 3 km/h.\n\nWalking distance is 9 km, so time = 9 / 3 = 3 hours.\n\nThen, add the coffee shop time t = 0.4 hours.\n\nTotal time = 3 + 0.4 = 3.4 hours\n\nConvert 3.4 hours to hours and minutes: 0.4 hours * 60 minutes/hour = 24 minutes.\n\nSo, total time is 3 hours and 24 minutes.\n\nBut the question asks for the number of minutes the walk takes, including t. So, 3 hours is 180 minutes, plus 24 minutes is 204 minutes.\n\nWait, but let me double-check that.\n\nAlternatively, 3.4 hours is 3 hours plus 0.4*60 = 24 minutes, so total is 3*60 +24=180+24=204 minutes.\n\nSo, the total time is 204 minutes.\n\nBut wait, let me check if I did all the steps correctly.\n\nFirst, found s = 2.5 km/h, correct.\n\nThen, t = 0.4 hours, correct.\n\nThen, new speed is 3 km/h, walking time is 3 hours, coffee shop time is 0.4 hours, total is 3.4 hours, which is 204 minutes.\n\nYes, that seems correct.\n\nBut let me make sure I didn't make any calculation errors.\n\nSo, s^2 + 2s - 11.25 = 0, solved s = 2.5.\n\nThen, walking time at 2.5 km/h: 9 / 2.5 = 3.6 hours.\n\nTotal time 4 hours, so t = 0.4 hours.\n\nAt s + 0.5 = 3 km/h, walking time is 9 / 3 = 3 hours.\n\nTotal time: 3 + 0.4 = 3.4 hours, which is 204 minutes.\n\nYes, that seems correct.\n</think>\n\nThe solution correctly calculates the walking speed and coffee shop time, then uses them to find the total time at the new speed. \n\nVerification: yes\n\nAnswer: \\boxed{204}"""
# def parse_verification(veri_res: str):
#     try:
#         PARSE_PATTERN = r"(?i)Verification[ \t]*:[ \t]*(Yes|No)"
#         match = re.search(PARSE_PATTERN, veri_res)
#         extracted_answer = match.group(1) if match else None
#         if extracted_answer.lower() == 'yes':
#             return 1
#         return 0
#     # all failures return false
#     except TypeError:
#         print("Error in extracting verification: {}".format(veri_res))
#         return 0
    
# print(parse_verification(res))

# print( """
# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

# Question: {question}""".format(question="11111"))
n = 32
row_data = {
        'model_path': 1,
        'dataset_name': 1,
        "TP": 2,
        "TN": 2,
        f"Maj_of_{n}": 'test',
}
print(row_data)