print("Starting LLUNK(Large Looping Universal Neural Kombiner), please wait...")
# Credit for original script goes to GJMorgan(LordGoonery) on Discord

import torch, shutil, json, concurrent.futures, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import psutil, os
import gradio as gr
import numpy as np
import random
from sys import argv
from tqdm import tqdm

# This should be a gradio numerical input, but I don't know gradio and this is an MVP
num_times_to_lunk = int(argv[1])


blend_ratio, fp16, always_output_fp16, max_shard_size, verbose_info, force_cpu, load_sharded = 0.5, False, True, "2000MiB", True, True, True
test_prompt, test_max_length = "Test, ", 32
blend_ratio_b = 1.0 - blend_ratio
def get_cpu_threads():
    try:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        return physical_cores, logical_cores
    except: return None
cpu_info = get_cpu_threads()
physical_cores, logical_cores = cpu_info if cpu_info else (4, 8)
def get_model_info(model):
    with torch.no_grad():
        outfo, cntent = "\n==============================\n", 0
        for name, para in model.named_parameters():
            cntent += 1
            outfo += ('{}: {}'.format(name, para.shape)) + "\n"
        outfo += ("Num Entries: " + str(cntent)) + "\n"
        outfo += ("==============================\n")
        return outfo

def merge_tensors(tensors, ratio_a, ratio_b):
    with torch.no_grad():
        for p1, p2 in tensors:
             if random.choice([True,False]) == True: # I should make a "replacement chance" variable (script argument) and use that instead of random.choice
                 p1 *= (blend_ratio  + np.random.random(0.1)[0]) # presumably the blend ratio arguments should be used here, but I'm leaving it unchanged for now since I just got this script and don't want to break it
                 p2 *= (blend_ratio_b  + np.random.random(0.1)[0])
                 p1 += p2
             else: p1, p2 = p1, p2


# Creates a list of lists of model parameters, split into num_parts parts, for parallel processing
def split_model_parameters(model, num_parts):
    parameters = list(model.parameters())
    part_size = (len(parameters) + num_parts - 1) // num_parts
    return [parameters[i:i + part_size] for i in range(0, len(parameters), part_size)]

# Merges two lists of model parameters, using the given blend ratio
def merge_models_part(part1, part2, ratio_a, ratio_b):
    merged_params = zip(part1, part2)
    valid_params = [(p1, p2) for p1, p2 in merged_params if p1.shape == p2.shape]
    merge_tensors(valid_params, ratio_a, ratio_b)

def merge_models(model1, model2, blend_ratio=0.6, num_threads=logical_cores):
    model1_params = split_model_parameters(model1, num_threads)
    model2_params = split_model_parameters(model2, num_threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for part1, part2 in zip(model1_params, model2_params):
            futures.append(executor.submit(merge_models_part, part1, part2, blend_ratio, 1.0 - blend_ratio))
        concurrent.futures.wait(futures)

def read_index_filenames(sourcedir):
    index = json.load(open(sourcedir + '/pytorch_model.bin.index.json', 'rt'))
    fl = [v for _, v in index['weight_map'].items()]
    return fl

def merge_models_and_save(model_path1, model_path2, model_path3=None):
    if not model_path1 or not model_path2:
        return "\nYou must select two directories containing models to merge and one output directory. Exiting."
    with torch.no_grad():
        if fp16:
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.float32)
        device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")

        # Why is the for loop above the model loading code? Because the merge code mutates model 1, so we need to clear it and load it again for each iteration of the loop
        for n in tqdm(range(num_times_to_lunk)): # note: num_times_to_lunk is extracted from sys.argv earlier in the script
            print("Loading Model 1...")
            model1 = AutoModelForCausalLM.from_pretrained(model_path1, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
            model1 = model1.to(device)
            model1.eval()
            print("Model 1 Loaded. Dtype: " + str(model1.dtype))
            print("Loading Model 2...")
            model2 = AutoModelForCausalLM.from_pretrained(model_path2, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
            model2 = model2.to(device)
            model2.eval()
            print("Model 2 Loaded. Dtype: " + str(model2.dtype))
            m1_info = get_model_info(model1)
            m2_info = get_model_info(model2)
            print("LUNKing models...")
            merge_models(model1, model2, blend_ratio)  # Pass the blend_ratio to merge_models function
            if model_path3:
                print("Saving new model...")
                newsavedpath = model_path3+f"/converted_model_{str(n)}"
                if always_output_fp16 and not fp16:
                    model1.half()
                model1.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
                print("\nSaved to: " + newsavedpath)
                # at this point, a bash command should be used to evaluate the model against the benchmark (run main.py in lm-evaluation-harness) and wait until it finishes and outputs its results to a file: ./{n}_results.json. The "waiting until it finishes" could be accomplished by checking if a file with the correct name exists every certain number of seconds, or it can be accomplished in a more clever way using libraries.
                cmd = f"python main.py --model hf-causal-experimental --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \ --tasks arc_challenge,hellaswag,truthfulqa --device cuda:0"
                if os.path.exists(model_path3+f"/converted_model_{str(n - 1)}"): # this line, or some modification of it, should be used to check if there is another version of the model to check against
                    pass # this here should check if the average of the accuracy of the results of the previous model are better than the current one, and if so, delete the current one and rename the previous one to have the current one's name
                    # else, the previous one is deleted
            else:
                print("\nOutput model was not saved as no output path was selected.")
                break

        print("\nScript Completed.")

current_directory = os.getcwd()

def interface(input_text1, input_text2,input_text3, blend_ratio_slider):  # Add the blend_ratio_slider parameter
    global blend_ratio
    blend_ratio = blend_ratio_slider  # Update the blend_ratio global variable
    merge_models_and_save(input_text1, input_text2, input_text3)
    return "Success! Models have been LUNKed."

iface = gr.Interface(
    fn=interface,
    inputs=[
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="FIRST model directory"),
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="SECOND model directory"),
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="OUTPUT model directory"),
        gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Blend Ratio")
    ],
    outputs="text",
    title="LLUNK(Large Looping Universal Neural Kombiner)",
    description="Select some models and mash them into a new one! Do this n times and pick the best one! So long as they're the same size and architecture..",
)

iface.launch()


cmd = f"python main.py --model hf-causal-experimental \
--model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
--tasks arc hellaswag,truthfulqa, \
--device cuda:0"
