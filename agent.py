from utils.config import Config
from utils.response import wrap_docs, wrap_ideas
from backend import FunctionSpec, query
from coder import CodeAgent
from idea_planner import Idea, IdeaSelector
from retriever import Retriever

import torch
import logging
import networkx as nx
import numpy as np
import os
import time

from typing import List, cast
from torch.multiprocessing import Queue, Manager
from rich.console import Console 
from rich.table import Table
from rich.live import Live

logger = logging.getLogger("graphml")

class Agent:
    def __init__(self, task_desc: str, cfg: Config):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.task_desc = task_desc

        assert torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"Num of detected gpus: {self.gpu_count}")

        self.kernels_retriever     = Retriever(cfg, cfg.doc_base_dir / "kernels")
        self.discussions_retriever = Retriever(cfg, cfg.doc_base_dir / "discussions")

        self.selector = IdeaSelector(
            ideas=[],
            feedback_model=self.acfg.feedback.model,
            feedback_temperature=self.acfg.feedback.temp,
        )
    
        logger.info("Agent initialized.")

    def get_initial_ideas(self, k: int = 4):
        docs = self.kernels_retriever.get_hotest_docs(k) + self.discussions_retriever.get_hotest_docs(k)
        self.summarize_docs(docs)

    def summarize_docs(self, docs: List[str]):
        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Your Task": (
                "I already have a list of ideas that partially explore how to approach this competition. Your job now is to:\n"
                "1. Read the following documents and summarize them in a few sentences.\n"
                "2. Break down these documents into individual and critical atomic ideas—these should be the smallest units of implementation (e.g., “use LightGBM for baseline,” “normalize input features,” “apply stratified K-fold CV”). Make sure each idea is indispensable for the final performance.\n"
            ),
            "Documents": wrap_docs(docs),
            "Instructions": {
                "Reminders": (
                    "1. Avoid using too general terms like 'Try different neural network architectures'\n"
                    "2. Focus on major techniques and methods that are inspiring\n"
                ),
                "Response Format": (
                    "Break down and collect atomic ideas and list them at the end of your response. You should avoid duplicated ideas. Format your output like this:\n"
                    "Your understanding of the task and a brief summarization of each document\n"
                    "Avoid repeating similar ideas. \n"
                    "===IDEA LIST===\n"
                    "-- atomic idea 1\n"
                    "-- atomic idea 2\n"
                    "-- atomic idea 3\n"
                    "...\n"
                ),
            }
        }

        response = query(
            system_message=prompt,
            user_message=None,
            model=self.acfg.feedback.model,
            temperature=self.acfg.feedback.temp,
        )

        for line in response.split("\n"):
            if line.startswith("--"):
                idea = line[2:].strip()
                logger.info(f"Found atomic idea: {idea}")
                self.add_idea(idea)

        return response

    def add_idea(self, content: str):
        idea = Idea(content)
        self.selector.add_idea(idea)

    def brainstorm(self):
        ideas = self.selector.get_ideas_str()

        logger.info(f"Brainstorming with ideas:\n{ideas}")
        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Your Task": (
                "I already have a list of ideas that partially explore how to approach this competition. Your job now is to:\n"
                "1. Think creatively and construct at least **3 alternative solution paths** that are likely to perform well, especially if combined with careful experimentation. \n"
                "2. Each solution path can be a strategy, pipeline, or method that combines multiple techniques. Try to make them as different as possible from the existing `ideas` list. \n"
                "3. After describing each full solution path, **break it down into individual atomic ideas**—these should be the smallest units of implementation (e.g., “use LightGBM for baseline,” “normalize input features,” “apply stratified K-fold CV”).\n"
                "4. Ensure these ideas do not substantially duplicate items already in `ideas`."
            ),
            "Ideas": ideas,
            "Instructions": {
                "Response Format": (
                    "Format your output like this:\n"
                    "<Your understanding of the task and explanation of your approaches>\n"
                    "===SOLUTION_PATH_1===\n"
                    "<Description of this approach>\n"
                    "-- atomic idea 1\n"
                    "-- atomic idea 2\n"
                    "-- atomic idea 3\n"
                    "...\n"
                    "===SOLUTION_PATH_2===\n"
                    "...\n"
                    "===SOLUTION_PATH_3===\n"
                    "...\n"
                ),
                "Reminder": "Be ambitious but realistic—many ideas can later be tested on a small subset of the data. Focus on novelty, diversity, and decomposability. Ready? Start."
            }
        }

        response = query(
            system_message=prompt,
            user_message=None,
            model=self.acfg.brainstorm.model,
            temperature=self.acfg.brainstorm.temp,
        )

        logger.info(f"Brainstorming response:\n{response}")

        for line in response.split("\n"):
            if line.startswith("--"):
                idea = line[2:].strip()
                logger.info(f"Found atomic idea: {idea}")
                self.add_idea(idea)

    def run(self):
        self.get_initial_ideas()

        for _ in range(self.acfg.iterations):
            self.brainstorm()
            logger.info("Brainstorming completed.")
            max_threads = min(self.gpu_count, self.acfg.max_threads)

            self.selector.reconstruct_ideas(max_ideas=self.acfg.max_ideas)

            selected_ideas = self.selector.select_subset(
                k=max_threads, 
                min_ideas_per_subset=self.acfg.min_ideas_per_subset,
                max_ideas_per_subset=self.acfg.max_ideas_per_subset
            )
            logger.info(f"Selected {max_threads} groups of ideas for implementation.")

            def render_table(status_dict):
                table = Table(title="Code Agents Status Panel", box=None, min_width=40)
                table.add_column("Agent ID", style="cyan", no_wrap=True)
                table.add_column("Status", style="magenta")

                for name, status in sorted(status_dict.items()):
                    table.add_row(str(name), status)

                return table

            with Manager() as manager:
                processes = []
                result_queue = Queue()
                status_dict = manager.dict()

                for i, indices in enumerate(selected_ideas):
                    ideas = self.selector.get_ideas_content_by_indices(indices)
                    logger.info(f"Running group {i + 1}/{max_threads} on GPU {i}...")
                    process = torch.multiprocessing.Process(
                        target=self.run_coder, 
                        args=(i, ideas, result_queue, status_dict)
                    )
                    process.start()
                    processes.append(process)
                
                console = Console()
                with Live(render_table(status_dict), console=console, refresh_per_second=2) as live:
                    while any(p.is_alive() for p in processes):
                        live.update(render_table(status_dict))
                        time.sleep(0.5)

                    # Final update
                    live.update(render_table(status_dict))
                    logger.info("\n✅ All code agents have completed!")

                for p in processes:
                    p.join()
                
                while not result_queue.empty():
                    device_id, result = result_queue.get()
                    self.selector.update_scores(selected_ideas[device_id], result)
        
        ### Final run
        logger.info("Final run...")
        selected_ideas = self.selector.select_subset(
            k=1,
            min_ideas_per_subset=self.acfg.min_ideas_per_subset,
            max_ideas_per_subset=self.acfg.max_ideas_per_subset
        )[0]
        ideas = self.selector.get_ideas_content_by_indices(selected_ideas)

        coder = CodeAgent(
            task_desc=self.task_desc, 
            cfg=self.cfg, 
            status_dict=None,
            is_final_run=True
        )

        coder.run(ideas, total_steps=20)
        
    def run_coder(self, device_id, ideas: List[str], result_queue: Queue, status_dict):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        status_dict[device_id] = "Initializing CodeAgent..."

        coder = CodeAgent(
            agent_id=device_id, 
            task_desc=self.task_desc, 
            cfg=self.cfg, 
            status_dict=status_dict
        )
        result = coder.run(ideas)

        assert len(result) == len(ideas)

        result_queue.put((device_id, result))




    

    