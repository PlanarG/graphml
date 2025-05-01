import backend
import random
import logging
import requests
import asyncio
import json

from utils.config import Config, load_cfg, prep_agent_workspace, load_task_desc
from retriever import Retriever
from executor import Executor
from typing import List, cast
from pathlib import Path
from uuid import uuid4
from omegaconf import OmegaConf
from backend import Conversation, FunctionSpec, query

from utils.response import extract_code, extract_text_up_to_code, wrap_code, wrap_docs, trim_long_string
from utils.metric import MetricValue, WorstMetricValue
from utils.data_preview import generate
from utils import get_timestamp

review_function_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (4-5 sentences) describing "
                " the empirical findings. If the code looks like an experiment, examine wheter its goals are achieved. "
                "Otherwise summarize the output and mention if the submission.csv was properly produced. "
                "Determine if the code is self-contained or some parts of the code are omitted. "
                " DO NOT suggest fixes or improvements.",
            },
            "output_abs": {
                "type": "string",
                "description": "select short and representative segments of the output log and mark the remainder as ellipses."
            }
        },
        "required": [
            "is_bug",
            "summary",
            "output_abs",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

review_function_spec_final = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (4-5 sentences) describing "
                " the empirical findings. If the code looks like an experiment, examine wheter its goals are achieved. "
                "Otherwise summarize the output and mention if the submission.csv was properly produced. "
                "Determine if the code is self-contained or some parts of the code are omitted. "
                " DO NOT suggest fixes or improvements.",
            },
            "output_abs": {
                "type": "string",
                "description": "select short and representative segments of the output log and mark the remainder as ellipses."
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully and produced submission.csv, report the value of the validation metric. Otherwise, leave it null."
            },
            "is_lower_better": {
                "type": "boolean", 
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy)."
            }
        },
        "required": [
            "is_bug",
            "summary",
            "output_abs",
            "metric",
            "is_lower_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

class CodeAgent:
    def __init__(
        self, 
        task_desc: str, 
        cfg: Config, 
        status_dict = None,
        agent_id: int = 0,
        is_final_run = False,
    ):
        self.task_desc    = task_desc
        self.agent_id     = agent_id
        self.cfg          = cfg
        self.step         = 0
        self.metric       = WorstMetricValue()
        self.acfg         = cfg.agent
        self.is_finished  = False
        self.status_dict  = status_dict
        self.is_final_run = is_final_run
        
        self.executor     = Executor(cfg.workspace_dir, timeout=cfg.exec.timeout, use_pty=True)
        self.conversation = Conversation(model=cfg.agent.code.model, temperature=cfg.agent.code.temp)

        response_file_name = "response_fmt_final.txt" if is_final_run else "response_fmt.txt"
        with open(Path(__file__).parent / response_file_name, "r") as f:
            self.response_fmt = f.read()    
        
        self.initialize_logger()
    
    def initialize_logger(self):
        log_format = "[%(asctime)s] %(levelname)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper()), format=log_format, handlers=[]
        )

        if self.is_final_run:
            logger = logging.getLogger(f"graphml-code-agent-final")
            log_dir = self.cfg.log_dir / f"coder-final"
        else:
            logger = logging.getLogger(f"graphml-code-agent-{self.agent_id}")
            log_dir = self.cfg.log_dir / f"coder-{self.agent_id}"

        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = get_timestamp()
        file_handler = logging.FileHandler(log_dir / f"{timestamp}.log")
        file_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(file_handler)

        self.logger = logger

    def initialize(self):
        prompt = {
            "Introduction": (
                "You're an expert Kaggle competitor tasked with implementing competition ideas into efficient Python code. "
                "The goal is to **evaluate all the ideas** we provide for you. "
            ),
            "Task Description": self.task_desc,
            "Data Overview": self.data_overview,
        }

        if self.is_final_run:
            prompt |= {
                "Available ideas": "\n".join([f"idea: {idea}" for idea in self.ideas]),
                "Instructions": {
                    "Response Format": self.response_fmt,
                    "Reminders": (
                        "- Avoid using progress bars in your code.\n"
                        "- **YOUR CODE MUST PRODUCE SUBMISSON AT ./working_final/submissions.csv, THIS IS EXTREMELY IMPORTANT**\n"
                        "- There is one A6000 gpu available for you, **maximize your use of computing resources**\n"
                        "- All you need to do is to implement these ideas and do hyperparameter tuning. You don't need to instroduce any new ideas or methods.\n"
                        "- You can use the './working_final' directory to store any temporary files that your code needs to create.\n"
                        "- Include at least one comment explaining your code. **No parts of the code should be skipped or omitted**, don't terminate before finishing the script.\n"
                        "- Remember, your ultimate goal is to **Obtain best score on this competition**. \n"
                        "- Your code should **print the value of the evaluation metric computed on a hold-out validation set.**\n"
                        "- Begin by summarizing your understanding of the task, and then choose your first action.\n"
                    )
                }
            }

        else:
            ideas_with_uuids = "\n".join([f"idea: {idea}, uuid: {uuid}" for idea, uuid in zip(self.ideas, self.uuids)])
            prompt |= {
                "Available ideas": ideas_with_uuids,
                "Available Options": (
                    "1. Propose your own code implementation (specify if it's for small-scale testing)\n"
                    "2. Summarize the progress made so far, evaluate each idea and quit\n"
                ),
                "Instructions": {
                    "Response Format": self.response_fmt,
                    "Reminders:": (
                        "- Avoid using progress bars in your code.\n"
                        "- Always wrap your action in one of the two markers above.\n"
                        "- For PROPOSE_CODE, include at least one comment explaining your code. **No parts of the code should be skipped or omitted**, don't terminate before finishing the script.\n"
                        "- To evaluate your implementation, you should print the value of the evaluation metric computed on a hold-out validation set.\n"
                        "- All the provided input data is stored in './input' directory.\n"
                        "- There is one A6000 gpu available for you, **maximize your use of computing resources**.\n"
                        f"- You should use the './working{self.agent_id}' directory to store any temporary files that your code needs to create. **Do not store data in the current ./ dir**. \n"
                        "- You should **use a subset (e.g., 10%) instead of full dataset** and test your code at a small scale at the beginning to avoid potential syntax errors or bugs. Given the very limited time available, please choose carefully if you wish to test on the full dataset. \n"
                        "- Remember, your ultimate goal is to **Provide a concrete evaluation of all ideas**. It's not necessary to obtain a best score or a perfect solution. You can only do some small experiments and observe the results.\n"
                        "- Begin by summarizing your understanding of the task, and then choose your first action.\n"
                    )
                }
            }

        prompt["Instructions"] |= self._prompt_environment

        self.logger.info(f"Init prompt: {json.dumps(prompt, indent=2)}")
        self.conversation.add_message(system_message=prompt)
    
    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    def set_status(self, status: str):
        if self.status_dict is not None:
            self.status_dict[self.agent_id] = f"Step {self.step}: {status}"

    def parse_exec_result(self, code_desc: str, code: str, exec_result) -> str:
        self.set_status("Analyzing the execution result...")

        output = exec_result['output']
        if exec_result['error'] == "Timeout":
            self.logger.info("Execution timed out.")
            output += "\n\n[System] Process was killed due to timeout.\n"

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if any part of the code is omitted, there were any bugs as well as report the empirical findings."
            )
        
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(code),
            "Goals and explanation": code_desc,
            "Execution output": wrap_code(trim_long_string(output), lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_function_spec_final if not self.is_final_run else review_function_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )

        result = (
            f"Terminal output (truncated): \n```\n{response['output_abs']}\n```\n"
            f"Execution summary: \n{response['summary']}\n"
        )

        if self.is_final_run:
            metric = response["metric"]
            submission_path = self.cfg.workspace_dir / "working_final" / "submissions.csv"

            if not isinstance(metric, (int, float)) or not submission_path.exists():
                metric = WorstMetricValue()
            else:
                metric = MetricValue(
                    response['metric'], maximize=not response['is_lower_better']
                )
            
            self.logger.info(f"Metric: {metric}")
            
            if metric > self.metric:
                self.logger.info(f"New best submission found.")

                self.metric = metric 
                best_submission_dir = self.cfg.workspace_dir / "best_submisson"
                best_submission_dir.mkdir(parents=True, exist_ok=True)
                submission_path.replace(best_submission_dir / "submissions.csv")

        return result

    def parse_summarization(self, response: str) -> dict:
        """Parse the response from the summarization step."""
        response = extract_code(response)
        lines = response.split("\n")
        scores = {}

        for line in lines:
            if ":" in line:
                uuid, score = line.split(":")
                assert uuid.strip() in self.uuids, f"UUID {uuid.strip()} not found in the list of UUIDs." 
                scores[uuid.strip()] = int(score.strip())
        
        for uuid in self.uuids:
            assert uuid in scores, f"UUID {uuid} not found in the scores."
            
        self.logger.info(f"Parsed scores: {scores}")

        return scores

    def query_and_parse_response(self):
        self.set_status("Querying the agent...")
        response = self.conversation.query()

        self.logger.info(f"Response: {response}")

        if "===PROPOSE_CODE===" in response:
            code = extract_code(response)
            code_desc = extract_text_up_to_code(response)
            code_desc = code_desc.replace("===PROPOSE_CODE===", "").replace("===END===", "").strip()

            if len(code) < 10 or code is None:
                self.logger.info("Code extraction failed. Prompting agent to retry.")
                summary = "Your response contains a format error. Please wrap your code inside a markdown code block. e.g. ```python\nprint('Hello World')\n```"
            else:
                self.set_status("Running code...")
                result = asyncio.run(self.executor.run(code, agent_file_name=f"agent_{self.agent_id}.py"))
                self.logger.info(f"Agent finished running the code with output \n{trim_long_string(result['output'])}\n.")

                summary = self.parse_exec_result(code_desc, code, result)

            self.logger.info(f"Execution summary: {summary}")

            if self.is_final_run:
                self.conversation.add_message(user_message=(
                    f"I ran your code and summarized the execution result:\n"
                    f"{summary}\n"
                    f"Now, please choose your next action using the same marker-based format as before. \n"
                    "A) Fix runtime/timeout errors (if any)\n"
                    "B) Do hyperparameter tuning\n"
                    "C) Include ideas that were not implemented yet\n"
                ))

            else:
                self.conversation.add_message(user_message=(
                    f"I ran your code and summarized the execution result:\n"
                    f"{summary}\n"
                    f"Now, please choose your next action using the same marker-based format as before. \n"
                    "A) Fix runtime errors (if any)\n"
                    "B) Iterate with adjusted parameters or code structures\n"
                    "C) Proceed with full dataset training\n"
                ))

        elif "===SUMMARIZE===" in response:
            try:
                self.scores = self.parse_summarization(response)
                self.set_status("Finished.")
                self.is_finished = True
            except Exception as e:
                self.logger.error(f"Error parsing the summarization response: {e}")
                self.conversation.add_message(user_message="Your summarization contains a format error. Please use the marker-based format response. You should evaluate all ideas provided. Do not contain any unnecessary content in the part surrounded by markdown code marks, each line of it should be uuid: score")

        else:
            self.logger.info("Response does not contain a valid action. Prompting agent to retry.")
            summary = "Your response contains a format error. Please use the marker-based format to indicate your next action."

            self.conversation.add_message(user_message=summary)
        
    def run(self, ideas: List[str], total_steps=10) -> List[int]:
        self.ideas = ideas
        self.uuids = [str(uuid4()) for _ in self.ideas]
        self.scores = None

        self.is_finished = False
        self.data_overview = generate(self.cfg.data_dir, include_file_details=True, simple=False)
        self.initialize()

        for step in range(total_steps):
            self.step = step
            self.query_and_parse_response()

            if self.is_finished:
                break
        
        if not self.is_final_run:
            while not self.is_finished:
                self.conversation.pop_message()
                self.conversation.add_message(user_message="Please summarize the results using the SUMMARIZE mark.")
                self.query_and_parse_response()
            
            assert self.scores is not None, "No scores found in the response."

            results = []
            for uuid in self.uuids:
                results.append(self.scores.get(uuid, None))
            
            
            return results
