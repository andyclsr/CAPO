# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor,as_completed
from time import sleep
from itertools import cycle 
import threading 

import torch
import re
import requests


from datetime import datetime


from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

BASE_URLS = [
    # put the address of nodes below
    # "http://xx.xx.xx.xx:30000", 
    # "http://xx.xx.xx.xx:30000", 
    # "http://xx.xx.xx.xx:30000", 

]

url_cycler = cycle(BASE_URLS)

url_lock = threading.Lock()

def get_next_url():
    with url_lock:
        return next(url_cycler)

API_KEY = "EMPTY"
MAX_RETRIES = 5
BASE_DELAY = 2
MODEL_NAME = "genrm-demo"
BATCH_SIZE = 384  # will be covered in func 'compute_score_batch'
TIMEOUT = 30000  

GENRM_PROMPT_TEMPLATE = """You are a math teacher.  Use [Ground Truth Solution] to find any erroneous step in [Solution To Judge]. And the solution to be checked is separated with steps like "<step_i>...</step_i>" where i is the index of the step. First, determine whether the solution is correct based on [Ground Truth Solution] and write it in the form "Judgement: Is the solution correct (Yes/No)? X",  where X is either Yes or No. If the solution is incorrect, review, analyze, and verify each step of the solution independently, without relying on the correctness of the context before. This means that the current step cannot be considered as incorrect simply because it used the conclusion derived from the previous erroneous reasoning. Then provide the list of the indices of error steps separated by a comma. And the format is as follows:\n <incorrect_steps>\n...(the index of the incorrect steps)...\n</incorrect_steps>.\n\n Please give your answer directly.\n\n[Question]\n{problem}\n\n[Ground Truth Solution]\n{gt_solution}\n\n[Solution To Judge]\n{solution}"""



def add_step_tags(text):
    steps = text.split('\n\n')
    
    tagged_steps = []
    for i, step in enumerate(steps):
        tagged_steps.append(f"<step_{i}>{step}</step_{i}>")
    
    return '\n\n'.join(tagged_steps)


def process_item(text):
    if text.startswith("<step_") and text.endswith(">"):
        index_str = text[6:-1]
        if index_str.isdigit():
            return int(index_str)
    else:
        if text.isdigit():
            return int(text)
    return None

def get_wrong_steps(text):
    pattern = r"<incorrect_steps>(.*?)</incorrect_steps>"
    match = re.search(pattern, text, re.DOTALL)

    extracted_list = []
    if match:
        content = match.group(1).strip()
        if content:
            items = content.split(',')
            for item in items:
                if item.strip() and process_item(item.strip()) is not None:
                    try:
                        extracted_list.append(int(process_item(item.strip())))
                    except:
                        print('{error', text,'}')
    else:
        pass
    return extracted_list


def split_steps(data, idx, is_llama):
    problem_length = len(data.batch['prompts'][idx])
    action_mask = data.batch['attention_mask'][idx, problem_length:]
    num_actions = len(action_mask)
    solution_tokens = data.batch['responses'][idx]

    # TODO here fix
    if not is_llama:
        # qwen: we prepare the tokens ended with \n\n in advance.
        split_step_tokens = torch.tensor([271, 382, 401, 555, 626, 630, 692, 1022, 1339, 1406, 1428, 1447, 1476, 1837, 1939, 2012, 2146, 2217, 2219, 2315, 2357, 2533, 2791, 2822, 2879, 3011, 3071, 3237, 3302, 3407, 3475, 3554, 3593, 3623, 3634, 3733, 3755, 3876, 4192, 4257, 4390, 4455, 4546, 4610, 4710, 4821, 5125, 5130, 5134, 5210, 5231, 5434, 5468, 5959, 6211, 6320, 6762, 7331, 7511, 7723, 7731, 8132, 8680, 8824, 9272, 9470, 9568, 9577, 9604, 10086, 10149, 10370, 10444, 10448, 10452, 11436, 11974, 12022, 12279, 12367, 12431, 12512, 12706, 12798, 13106, 13246, 14223, 14333, 14512, 14599, 14621, 14711, 14731, 14808, 14929, 15047, 15075, 15424, 15436, 15441, 15483, 15514, 15538, 15620, 15674, 15766, 15799, 16117, 16218, 16968, 17199, 17477, 17701, 17745, 18259, 18292, 18459, 18507, 18544, 18556, 18611, 18797, 19144, 19235, 19324, 19328, 19347, 19421, 19513, 19799, 19896, 20225, 20356, 20375, 20707, 21174, 21238, 21518, 21613, 21668, 21675, 21696, 21702, 21906, 21974, 22116, 22525, 22663, 22701, 22712, 22746, 23398, 23459, 23754, 24197, 24391, 24616, 24727, 24750, 24796, 24825, 25138, 25162, 25321, 25464, 25501, 25571, 25639, 25897, 25912, 26469, 26487, 26578, 26850, 27113, 27126, 27311, 27352, 27427, 27701, 27771, 27818, 27866, 27901, 28038, 28075, 28348, 28372, 28389, 28429, 28581, 29084, 29122, 29184, 29562, 29636, 30034, 30458, 30463, 30625, 30831, 31225, 31295, 31307, 31483, 31707, 31797, 32057, 32423, 32623, 32636, 32805, 33351, 33498, 33621, 33666, 33694, 33862, 33933, 34149, 34184, 34332, 34499, 34583, 34773, 34985, 35115, 35184, 35219, 35721, 35786, 35829, 35833, 36928, 36979, 36984, 37859, 38225, 38497, 39024, 39365, 39774, 39865, 39887, 39974, 40254, 40725, 40901, 41025, 41037, 41401, 41620, 41843, 42015, 42273, 42450, 43060, 43153, 43501, 43608, 43738, 44316, 44360, 44364, 44611, 44651, 44732, 44993, 45100, 45128, 45320, 45806, 46644, 46739, 46796, 47144, 47446, 47449, 47486, 47989, 48320, 48443, 48622, 48962, 49088, 49270, 49555, 49760, 49962, 50524, 50640, 50940, 50970, 51030, 51278, 51308, 51418, 51632, 51754, 52054, 52294, 52324, 52338, 52599, 53099, 53505, 53589, 53632, 54060, 54210, 55144, 55266, 55342, 55430, 55919, 55957, 56141, 56177, 56761, 56831, 56870, 56948, 56993, 57073, 57351, 57475, 57545, 57570, 57777, 57944, 58071, 58157, 58177, 58375, 58418, 58501, 58606, 58629, 58724, 58872, 58935, 59454, 59480, 59482, 59581, 59610, 59841, 59928, 60288, 60460, 60543, 60803, 60998, 61163, 61277, 61439, 61657, 61827, 61969, 62021, 62338, 62610, 62877, 62965, 63159, 63966, 64139, 64277, 64329, 64631, 65052, 65225, 65264, 65267, 65579, 65668, 65887, 65974, 66371, 66376, 66426, 66506, 66786, 66816, 66929, 66960, 67436, 67564, 67625, 67864, 67940, 67977, 68013, 68101, 68303, 68327, 68562, 68601, 68612, 68786, 69043, 69178, 69493, 69494, 69911, 70031, 70180, 70191, 70193, 70571, 70652, 70674, 70818, 71146, 71248, 71496, 71515, 71537, 71612, 72103, 72229, 72389, 72648, 72764, 72931, 73330, 73530, 73663, 73822, 73900, 73952, 73953, 74123, 74203, 74376, 74384, 74385, 74525, 74526, 74687, 75048, 75258, 75499, 75694, 75719, 75743, 75884, 75910, 75960, 76058, 76325, 76379, 78137, 78241, 78314, 78435, 78900, 78929, 78988, 79083, 79226, 79279, 79304, 79322, 79364, 79483, 79515, 79567, 79739, 79931, 80394, 80634, 80823, 80874, 80984, 81142, 81392, 81436, 81452, 81645, 81767, 81892, 82216, 82361, 83007, 83775, 83809, 83900, 84025, 84500, 84741, 85243, 85312, 85321, 85545, 85604, 85617, 86197, 86214, 86779, 86853, 87008, 87036, 87079, 87141, 87248, 87346, 87567, 87586, 87736, 87894, 87951, 88252, 88946, 89238, 89411, 89684, 89720, 89958, 90222, 90408, 91545, 91581, 91697, 91737, 91935, 92108, 92327, 92346, 92379, 92855, 92986, 93045, 93047, 93244, 93490, 93596, 93611, 93670, 93682, 93718, 93873, 93902, 93922, 94081, 94141, 94345, 94367, 94464, 94521, 94799, 95173, 95377, 96332, 97833, 98140, 98320, 98372, 98422, 98527, 98788, 98965, 98973, 99058])
    else:
        # llama: we prepare the tokens ended with \n\n in advance.
        split_step_tokens = torch.tensor([271, 382, 401, 557, 629, 633, 696, 1038, 1363, 1432, 1454, 1473, 1504, 1875, 1980, 2055, 2195, 2266, 2268, 2368, 2412, 2595, 2861, 2892, 2950, 3086, 3147, 3317, 3382, 3490, 3559, 3638, 3677, 3707, 3718, 3818, 3840, 3961, 4286, 4352, 4489, 4555, 4649, 4713, 4815, 4926, 5235, 5240, 5244, 5322, 5344, 5551, 5585, 6087, 6343, 6454, 6905, 7481, 7663, 7879, 7887, 8295, 8851, 9000, 9456, 9658, 9763, 9772, 9801, 10294, 10359, 10586, 10661, 10665, 10669, 11690, 12241, 12291, 12559, 12647, 12713, 12795, 12996, 13090, 13407, 13549, 14557, 14670, 14852, 14941, 14963, 15053, 15073, 15152, 15276, 15397, 15425, 15786, 15799, 15804, 15850, 15882, 15908, 15993, 16049, 16143, 16176, 16508, 16616, 17398, 17642, 17935, 18171, 18218, 18760, 18796, 18966, 19014, 19053, 19066, 19124, 19327, 19691, 19789, 19884, 19888, 19908, 19985, 20083, 20386, 20490, 20838, 20979, 20999, 21366, 21863, 21932, 22242, 22341, 22406, 22414, 22438, 22445, 22669, 22742, 22896, 23341, 23494, 23535, 23547, 23584, 24287, 24356, 24688, 25174, 25393, 25638, 25758, 25782, 25833, 25863, 26196, 26221, 26389, 26543, 26582, 26652, 26722, 26986, 27001, 27567, 27585, 27676, 27948, 28212, 28225, 28411, 28452, 28527, 28801, 28871, 28918, 28966, 29001, 29138, 29175, 29448, 29472, 29489, 29529, 29681, 30184, 30222, 30284, 30662, 30736, 31134, 31558, 31563, 31725, 31931, 32325, 32395, 32407, 32583, 32807, 32897, 33157, 33523, 33723, 33736, 33905, 34451, 34598, 34721, 34766, 34794, 34962, 35033, 35249, 35284, 35432, 35599, 35683, 35873, 36085, 36215, 36284, 36319, 36821, 36886, 36929, 36933, 38028, 38079, 38084, 38959, 39325, 39597, 40124, 40465, 40874, 40965, 40987, 41074, 41354, 41825, 42001, 42125, 42137, 42501, 42720, 42943, 43115, 43373, 43550, 44160, 44253, 44601, 44708, 44838, 45416, 45460, 45464, 45711, 45751, 45832, 46093, 46200, 46228, 46420, 46906, 47744, 47839, 47896, 48244, 48546, 48549, 48586, 49089, 49420, 49543, 49722, 50062, 50188, 50370, 50655, 50860, 51062, 51624, 51740, 52040, 52070, 52130, 52378, 52408, 52518, 52732, 52854, 53154, 53394, 53424, 53438, 53699, 54199, 54605, 54689, 54732, 55160, 55310, 56244, 56366, 56442, 56530, 57019, 57057, 57241, 57277, 57861, 57931, 57970, 58048, 58093, 58173, 58451, 58575, 58645, 58670, 58877, 59044, 59171, 59257, 59277, 59475, 59518, 59601, 59706, 59729, 59824, 59972, 60035, 60554, 60580, 60582, 60681, 60710, 60941, 61028, 61388, 61560, 61643, 61903, 62098, 62263, 62377, 62539, 62757, 62927, 63069, 63121, 63438, 63710, 63977, 64065, 64259, 65066, 65239, 65377, 65429, 65731, 66152, 66325, 66364, 66367, 66679, 66768, 66987, 67074, 67471, 67476, 67526, 67606, 67886, 67916, 68029, 68060, 68536, 68664, 68725, 68964, 69040, 69077, 69113, 69201, 69403, 69427, 69662, 69701, 69712, 69886, 70143, 70278, 70593, 70594, 71011, 71131, 71280, 71291, 71293, 71671, 71752, 71774, 71918, 72246, 72348, 72596, 72615, 72637, 72712, 73203, 73329, 73489, 73748, 73864, 74031, 74430, 74630, 74763, 74922, 75000, 75052, 75053, 75223, 75303, 75476, 75484, 75485, 75625, 75626, 75787, 76148, 76358, 76599, 76794, 76819, 76843, 76984, 77010, 77060, 77158, 77425, 77479, 79237, 79341, 79414, 79535, 80000, 80029, 80088, 80183, 80326, 80379, 80404, 80422, 80464, 80583, 80615, 80667, 80839, 81031, 81494, 81734, 81923, 81974, 82084, 82242, 82492, 82536, 82552, 82745, 82867, 82992, 83316, 83461, 84107, 84875, 84909, 85000, 85125, 85600, 85841, 86343, 86412, 86421, 86645, 86704, 86717, 87297, 87314, 87879, 87953, 88108, 88136, 88179, 88241, 88348, 88446, 88667, 88686, 88836, 88994, 89051, 89352, 90046, 90338, 90511, 90784, 90820, 91058, 91322, 91508, 92645, 92681, 92797, 92837, 93035, 93208, 93427, 93446, 93479, 93955, 94086, 94145, 94147, 94344, 94590, 94696, 94711, 94770, 94782, 94818, 94973, 95002, 95022, 95181, 95241, 95445, 95467, 95564, 95621, 95899, 96273, 96477, 97432, 98933, 99240, 99420, 99472, 99522, 99627, 99888, 100065, 100073, 100158, 102175, 102239, 102276, 104480, 104881, 105291, 106959, 108949, 109398, 111496, 115205, 115684, 118599, 118995, 119029, 122886, 125179, 125793, 125837, 126595, 127008, 127140, 127420])
    # find step separator, typically '\n\n'
    column_ids = torch.where(
        torch.isin(solution_tokens, split_step_tokens)
    )
    # +1 for the last step with eos instead of step separator
    max_num_steps = column_ids[0].numel() + 2
    # end index of each step, shape: (B, max_num_steps), type: long
    score_ids = torch.full(
        (max_num_steps,), 0, dtype=torch.long, 
    )
    # whether end of step, shape: (B, max_response_tokens), type: bool
    action_mask = action_mask.unsqueeze(0)
    eos_indice = num_actions - 1 - action_mask.long().fliplr().argmax(1)

    # intermediate steps
    score_ids[1:max_num_steps-1] = column_ids[0]
    # last step
    score_ids[max_num_steps-1] = eos_indice
    
    # score_ids, score_mask, reward_mask for data.batch['responses'],
    output = dict(
        step_max=max_num_steps-2,
        step_positions=score_ids,
    )
    return output

def get_response(prompt_lst):
    """
    using api to get a batch responses
    """
    for attempt in range(MAX_RETRIES):
        try:
            base_url = get_next_url()
            headers = {"Content-Type": "application/json"}
            chat_url = f"{base_url}/v1/completions"
            data = {"model": MODEL_NAME, "prompt": prompt_lst, "max_tokens": 512}
            output = requests.post(chat_url, headers=headers, json=data, timeout=TIMEOUT)
            output.raise_for_status()
            responses = [choice["text"] for choice in output.json()["choices"]]
            print(f"temp debug: {repr(responses[0])}")
            return responses
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Exception from {base_url}: {repr(e)}")
                delay = BASE_DELAY * (2**attempt)
                sleep(delay)
                print(f"Retrying in {delay} seconds...")
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error from {base_url}: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for batch starting with prompt '{prompt_lst[0]}' after all retries!")


def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score




def compute_score(response, solution_str, data, idx, rule_based_reward, reward_kwargs):
    """
    deal with a single response and compute reward score tensor.
    """

    wrong_step_indices = get_wrong_steps(response)
    
    reward_tensor = torch.full((data.batch['responses'][idx].shape[-1],), rule_based_reward * reward_kwargs['sentence_weight'], dtype=torch.float32, device=data.batch["responses"][idx].device)

    output = split_steps(data, idx, reward_kwargs.get('is_llama', False))

    if output['step_max'] != solution_str.count('<step_')-1:
        print(f"Warning: step split mismatch, expected {solution_str.count('<step_')}, got {output['step_max']}")
        return reward_tensor

    step_positions = output['step_positions']
    max_num_steps = output['step_max']
    for step_idx in wrong_step_indices[:reward_kwargs['max_wrong_steps']]:
        if step_idx <= max_num_steps:
            reward_tensor[step_positions[step_idx]:step_positions[step_idx+1]] += -1 * reward_kwargs['process_weight']

    return reward_tensor


def compute_score_batch(data_source, solution_str, ground_truth, extra_info, data, **reward_kwargs):
    if isinstance(extra_info, dict):
        ck = extra_info
    else:
        ck = extra_info[0]

    from verl.utils.reward_score import default_compute_score
    rule_based_reward = []
    reward_return = []
    
    for data_s, sol_str, gt, extra_if in zip(
        data_source, solution_str, ground_truth, extra_info, strict=True
    ):
        score = default_compute_score(data_s, sol_str, gt, extra_if)
        rule_based_reward.append(score['score']) 
        reward_return.append(score)

    if ck["split"] == "test":
        return reward_return

    prompts_to_process = []
    tagged_solutions = []
    for sol_str, extra_if, gt in zip(solution_str, extra_info, ground_truth, strict=True):
        problem = extra_if["question"]
        tagged_sol = add_step_tags(sol_str)
        tagged_solutions.append(tagged_sol)
        is_orm = reward_kwargs.get('genorm', False)
        prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=tagged_sol, gt_solution=gt)

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_to_process.append(text)

    
    BATCH_SIZE = min(len(prompts_to_process) // len(BASE_URLS), 384)
    print(f"Total prompts to process: {len(prompts_to_process)}. Batch size: {BATCH_SIZE}.")
    all_batches = [prompts_to_process[i:i + BATCH_SIZE] for i in range(0, len(prompts_to_process), BATCH_SIZE)]
    print(f"Split into {len(all_batches)} batches.")

    all_responses = []
    

    num_workers = len(BASE_URLS) 
    print(f"[{begin_t}] Starting ThreadPoolExecutor with {num_workers} workers to send batches in parallel...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch_index = {executor.submit(get_response, batch): i for i, batch in enumerate(all_batches)}
        
        results_in_order = [None] * len(all_batches)

        for future in as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            try:
                responses_batch = future.result()
                results_in_order[batch_index] = responses_batch
                print(f"  > Successfully received results for batch {batch_index + 1}/{len(all_batches)}")
            except Exception as e:
                print(f"  ! Batch {batch_index + 1}/{len(all_batches)} failed permanently after all retries: {e}")
                failed_batch_size = len(all_batches[batch_index])
                results_in_order[batch_index] = [""] * failed_batch_size
    
    for batch_result in results_in_order:
        all_responses.extend(batch_result)
    
    results = []
    for idx, response in enumerate(all_responses):
        result = compute_score(
            response, 
            tagged_solutions[idx],
            data, 
            idx, 
            rule_based_reward[idx], 
            reward_kwargs
        )
        results.append(result)

    return results