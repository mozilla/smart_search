import os
import time
from retrieval import get_normalized_name
from datasets import load_dataset


start_time = time.time()

def get_all_profiles(hf_repo):
    docs = load_dataset(hf_repo, name="docs", download_mode="force_redownload")["train"]
    unique_profiles = sorted(set(docs["profile"]))
    return unique_profiles

if __name__ == "__main__":

    hf_repo = "frankjc2022/semantic-history-search"

    model_names = ["Xenova/all-MiniLM-L6-v2", "minishlab/potion-retrieval-32M"]

    # profiles = [
    #     "025_creative_hobbyist"
    # ]
    profiles = get_all_profiles(hf_repo)


    cwd = os.getcwd()


    c = 0
    for profile in profiles:
        c += 1
        print(f"========== {c:02}/{len(profiles):02} Working on profile: {profile} ==========")
        save_path = f"results/{profile}"
        for model_name in model_names:
            print(f"=== Working on model: {model_name} ===")
            cmd = f"""
            python retrieval.py \
                --model_name "{model_name}" \
                --save_path {save_path} \
                --profile {profile} \
                --hf_repo {hf_repo}
            """
            os.system(cmd)

            result_filename = f"{save_path}/result__{get_normalized_name(model_name)}.csv"
            cmd = f"""python evaluation.py -f {os.path.join(cwd, result_filename)} --save_path {save_path}"""
            os.system(cmd)
        print(f"Done profile {profile} ({c:02}/{len(profiles):02}): {time.time() - start_time:.0f} seconds")

    print("="*100)
    cmd = f"""python generate_plots.py --base_dir results"""
    os.system(cmd)
    cmd = f"""python build_report.py --base_dir results --out_html results/eval_report.html"""
    os.system(cmd)