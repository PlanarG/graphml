from utils.response import wrap_docs, wrap_ideas
from backend import FunctionSpec, query

import torch
import logging
import networkx as nx
import numpy as np

from ast import literal_eval
from dataclasses import dataclass
from uuid import uuid4
from typing import List, cast
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

logger = logging.getLogger("graphml")

@dataclass
class Idea:
    content: str
    total_scores: int = 0
    num_tries: int = 0

    def get_avg_score(self) -> float:
        if self.num_tries == 0:
            return 0.0
        return self.total_scores / self.num_tries

    def __str__(self):
        avg_score = "N/A" if self.num_tries == 0 else f"{self.total_scores / self.num_tries:.2f}"
        return f"Idea: {self.content}; Average score: {avg_score} (out of 10); Number of tries: {self.num_tries} "

    def __post_init__(self):
        self.uuid = uuid4()
        self.ucb_score = None

class IdeaSelector:
    def __init__(
        self, 
        ideas: List[Idea], 
        feedback_model: str,
        feedback_temperature: float,
        alpha: float = 0.7,
    ):
        self.feedback_model = feedback_model
        self.feedback_temperature = feedback_temperature
        self.ideas = ideas
        self.alpha = alpha
    
    def add_idea(self, idea: Idea):
        self.ideas.append(idea)

    def get_ideas_str(self):
        return wrap_ideas([str(idea) for idea in self.ideas])

    def get_ideas_content_by_indices(self, indices: List[int]):
        return [self.ideas[i].content for i in indices]
    
    def update_scores(self, idea_indices: List[int], scores: List[float]):
        assert len(idea_indices) == len(scores), "Idea indices and scores must have the same length."

        scores = np.array(scores)
        scores_avg = np.mean(scores)
        scores -= scores_avg
        scores = scores.tolist()

        for i, score in enumerate(scores):
            self.ideas[idea_indices[i]].num_tries += 1
            self.ideas[idea_indices[i]].total_scores += score

            logger.info(
                f"Updated idea {self.ideas[idea_indices[i]].content} with score {score}. "
            )

    
    def reconstruct_ideas(self, max_ideas: int, max_tries: int = 3):
        assert len(self.ideas) > 0, "Ideas list cannot be empty."

        if len(self.ideas) > max_ideas:
            logger.info(f"Discarding ideas to keep only {max_ideas} ideas.")
            sorted_ideas = sorted(self.ideas, key=lambda x: x.get_avg_score(), reverse=True)
            self.ideas = sorted_ideas[:max_ideas]
        
        function_spec = FunctionSpec(
            name="reconstruct_ideas",
            json_schema={
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "string",
                        "description": "required format: python list of tuples, each tuple contains three elements: the description of the idea, the total score, and the number of tries",
                    },
                },
                "required": [
                    "ideas",
                ],
            },
            description="Decompose, merge, and reconstruct the ideas",
        )

        ideas_str = wrap_ideas([str(idea) for idea in self.ideas])
        prompt = (
            "You are a machine learning expert. After carefully searching the relevant literature, you have come up with a list of ideas to implement. However, this idea list has some issues: \n"
            "- Some ideas are too similar and should be merged into one. \n"
            "- Some ideas are too complex and should be decomposed. \n"
            "- Some ideas are overlapping, you should rephrase and decouple them. \n"
            "- You should discard ideas that are irrelevant to the final performance, such as error visualization, etc. \n"
            "Each idea has three properties: idea description, total score over all tries, and number of tries. If you merge two ideas, the total score and number of tries should be the sum of the two ideas. If you decompose one idea into multiple minor ideas, they should inherit the idea's total score and number of tries. \n"
            "Please decompose, merge, and reconstruct the ideas. \n"
            f"{ideas_str}"
        )

        for _ in range(max_tries):
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=function_spec,
                    model=self.feedback_model,
                    temperature=self.feedback_temperature,
                )
            )

            if response["ideas"] is not None:
                try:
                    ideas_list = literal_eval(response["ideas"])
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Error parsing ideas list: {e}")
                    continue

                try:
                    self.ideas = []
                    for idea in ideas_list:
                        content, total_scores, num_tries = idea
                        assert isinstance(content, str), "Idea content must be a string."
                        assert isinstance(total_scores, (int, float)), "Total scores must be a number."
                        assert isinstance(num_tries, int), "Number of tries must be an integer."
                        self.ideas.append(Idea(content, total_scores, num_tries))

                except (ValueError, AssertionError) as e:
                    logger.warning(f"Error processing ideas list: {e}")
                    continue
                
                break

            else:
                logger.warning("No ideas list found in the response. Retrying...")

    def _build_conflict_graph(
        self,
        ideas: List[Idea], 
        max_retries: int = 3
    ) -> nx.Graph:
        logger.info("Building conflict graph...")

        G = nx.Graph()
        G.add_nodes_from(range(len(ideas)))

        ideas_str = wrap_ideas([str(idea) for idea in ideas]) 

        for _ in range(max_retries):
            function_spec = FunctionSpec(
                name="submit_conflict_ideas",
                json_schema={
                    "type": "object",
                    "properties": {
                        "conflict_list": {
                            "type": "string",
                            "description": "required format: python list of tuples, each tuple contains two ideas' indices",
                        },
                    },
                    "required": [
                        "conflict_list",
                    ],
                },
                description="Determine all conflicts between any two ideas",
            )

            prompt = (
                "We have a list of ideas and want to implement them into a single python code. However, some ideas are different approaches to the same topic and are not appropriate to be applied simultaneously. \n"
                "Moreover, if two ideas are too similar, you should consider them as conflicting. \n"
                "Please determine all conflicts between any two ideas. \n"
                f"{ideas_str}"
            )   

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=function_spec,
                    model=self.feedback_model,
                    temperature=self.feedback_temperature,
                )
            )

            if response["conflict_list"] is not None:
                try:
                    conflict_list = literal_eval(response["conflict_list"])
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Error parsing conflict list: {e}")
                    continue

                for conflict in conflict_list:
                    i, j = conflict
                    logger.info(f"Adding edge between ideas {i} and {j} due to conflict.")
                    G.add_edge(i, j)
                
                break

            else:
                logger.warning("No conflict list found in the response. Retrying...")

        return G
    
    def select_subset(
        self, 
        k: int,
        min_ideas_per_subset: int = 4,
        max_ideas_per_subset: int = 8
    ) -> List[List[int]]:
        assert len(self.ideas) > 0, "Ideas list cannot be empty."

        self.conflict_graph = self._build_conflict_graph(self.ideas)

        for i in range(len(self.ideas)):
            logger.info(f"Node {i}: {self.ideas[i].content} {self.ideas[i].get_avg_score()}")

        vertices = list(self.conflict_graph.nodes())
        weights = {v: self.ideas[v].get_avg_score() for v in vertices}

        logger.info(f"Finding top {k} weighted independent sets...")

        prob = LpProblem(f"Maximize_Weight", LpMaximize)

        ### Variables

        x = []
        for i in range(k):
            # x[i][j] = 1 if vertex j is in the i-th independent set
            x.append({v: LpVariable(f"x_{i}_{v}", cat="Binary") for v in vertices})   

        z = []
        for i in range(k):
            z_i = [[]] * k
            for j in range(i + 1, k):
                # z[i][j][v] = 1 if vertex v is in joint of the i-th independent set and j-th independent set
                z_i_j = {v: LpVariable(f"z_{i}_{j}_{v}", cat="Binary") for v in vertices}
                z_i[j] = z_i_j
            z.append(z_i)
        
        ### Objective

        w_expr = lpSum(weights[v] * x[i][v] for i in range(k) for v in vertices)
        joint_expr = lpSum(z[i][j][v] for i in range(k) for j in range(i + 1, k) for v in vertices)

        prob += w_expr - self.alpha * joint_expr

        ###  Constraints

        for i in range(k):
            for u, v in self.conflict_graph.edges():
                prob += x[i][u] + x[i][v] <= 1
        
        for v in vertices:
            for i in range(k):
                for j in range(i + 1, k):
                    prob += x[i][v] + x[j][v] - z[i][j][v] <= 1
        
        for i in range(k):
            prob += lpSum(x[i][v] for v in vertices) >= min_ideas_per_subset
            prob += lpSum(x[i][v] for v in vertices) <= max_ideas_per_subset
        
        # prob.solve(PULP_CBC_CMD(msg=False))
        prob.solve()

        if LpStatus[prob.status] != "Optimal": 
            logger.error(f"No optimal solution found. Stopping.")
            return []
            
        result = [[v for v in vertices if x[i][v].value() == 1] for i in range(k)]
        return result
