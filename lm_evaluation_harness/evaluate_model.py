import argparse
import json
import logging
import os

from lm_eval import evaluator, utils
from lm_eval import tasks as eval_tasks

logging.getLogger("openai").setLevel(logging.WARNING)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--model_args", default="")
#     parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)) # for the purposes of this test, tasks should = arc hellaswag mmlu truthfulqa hendrycks_ethics hendrycks_math hendrycks_test
#     parser.add_argument("--provide_description", action="store_true")
#     parser.add_argument("--num_fewshot", type=int, default=0)
#     parser.add_argument("--batch_size", type=str, default=None)
#     parser.add_argument("--max_batch_size", type=int, default=None,
#                         help="Maximal batch size to try with --batch_size auto")
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--output_path", default=None)
#     parser.add_argument("--limit", type=float, default=None,
#                         help="Limit the number of examples per task. "
#                              "If <1, limit is a percentage of the total number of examples.")
#     parser.add_argument("--data_sampling", type=float, default=None)
#     parser.add_argument("--no_cache", action="store_true")
#     parser.add_argument("--decontamination_ngrams_path", default=None)
#     parser.add_argument("--description_dict_path", default=None)
#     parser.add_argument("--check_integrity", action="store_true")
#     parser.add_argument("--write_out", action="store_true", default=False)
#     parser.add_argument("--output_base_path", type=str, default=None)

#     return parser.parse_args()


def evaluate_model(model=None, model_args="", tasks=None, provide_description=False, num_fewshot=0, batch_size=None, max_batch_size=None, device=None, output_path=None, limit=None, data_sampling=None, no_cache=False, decontamination_ngrams_path=None, description_dict_path=None, check_integrity=False, write_out=False, output_base_path=None):
    # args = parse_args()

    # if limit:
    #     print(
    #         "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
    #     )

    # if tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    task_names = utils.pattern_match(tasks.split(","), eval_tasks.ALL_TASKS)
    print("\n\n\n====================")
    print(os.getcwd())
    print("\n\n\n====================")

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if description_dict_path:
        with open(description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        no_cache=no_cache,
        limit=limit,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        check_integrity=check_integrity,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dumped)
    else:
        print("WARNING: accuracy results not saved anywhere, specify an output path with --output_path")
    return results