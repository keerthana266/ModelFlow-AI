# app/ml/transpiler.py

from typing import Any, Dict, List

from app.ml.trainer import train_automl
from app.services.storage import get_dataset_path


def normalize_node_type(node_type: str) -> str:
    return str(node_type or "").strip().lower()


def find_root_node(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    target_nodes = {edge["target"] for edge in edges}
    root_nodes = [node["id"] for node in nodes if node["id"] not in target_nodes]

    if not root_nodes:
        raise ValueError("Invalid pipeline: no starting node found.")

    if len(root_nodes) > 1:
        raise ValueError("Invalid pipeline: only one starting node is supported.")

    return root_nodes[0]


def find_next_node_id(current_node_id: str, edges: List[Dict[str, Any]]) -> str | None:
    outgoing_edges = [edge for edge in edges if edge["source"] == current_node_id]

    if len(outgoing_edges) > 1:
        raise ValueError("Branching pipelines are not supported yet.")

    if not outgoing_edges:
        return None

    return outgoing_edges[0]["target"]


def execute_pipeline_graph(nodes: list, edges: list) -> dict:
    if not nodes:
        raise ValueError("Pipeline must contain at least one node.")

    node_map = {node["id"]: node for node in nodes}

    current_node_id = find_root_node(nodes, edges)
    visited_nodes = set()

    pipeline_state: Dict[str, Any] = {}
    execution_log = []

    while current_node_id:
        if current_node_id in visited_nodes:
            raise ValueError("Invalid pipeline: cycle detected.")

        visited_nodes.add(current_node_id)

        node = node_map.get(current_node_id)

        if not node:
            raise ValueError(f"Node '{current_node_id}' was not found.")

        node_type = normalize_node_type(node.get("type"))
        node_data = node.get("data", {}) or {}

        execution_log.append(f"Executing node: {node_type} ({current_node_id})")

        if node_type == "dataloader":
            dataset_id = node_data.get("dataset_id")
            target_column = node_data.get("target_column")

            if not dataset_id:
                raise ValueError("Data Loader node is missing dataset_id.")

            if not target_column:
                raise ValueError("Data Loader node is missing target_column.")

            dataset_path = get_dataset_path(dataset_id)

            pipeline_state["dataset_id"] = dataset_id
            pipeline_state["dataset_path"] = str(dataset_path)
            pipeline_state["target_column"] = target_column

            execution_log.append(f"-> Dataset found: {dataset_id}")
            execution_log.append(f"-> Target column: {target_column}")

        elif node_type == "automl":
            dataset_path = pipeline_state.get("dataset_path")
            target_column = pipeline_state.get("target_column")

            if not dataset_path or not target_column:
                raise ValueError("AutoML node cannot run before Data Loader node.")

            time_budget = int(node_data.get("time_budget", 30))
            time_budget = max(10, min(time_budget, 300))

            tuning_mode = node_data.get("tuning_mode", "flaml")

            if tuning_mode not in ["flaml", "optuna"]:
                raise ValueError("Invalid tuning_mode. Use 'flaml' or 'optuna'.")

            execution_log.append(f"-> Triggering AutoML training for target: {target_column}")
            execution_log.append(f"-> Training mode: {tuning_mode}")
            execution_log.append(f"-> Time budget: {time_budget} seconds")

            task = train_automl.delay(
                dataset_path,
                target_column,
                time_budget,
                tuning_mode,
            )

            pipeline_state["task_id"] = task.id
            pipeline_state["model_status"] = "Training Initiated"
            pipeline_state["time_budget"] = time_budget
            pipeline_state["tuning_mode"] = tuning_mode

            execution_log.append(f"-> Background Celery task queued: {task.id}")

        else:
            execution_log.append(f"-> Unknown node type '{node_type}' ignored.")

        current_node_id = find_next_node_id(current_node_id, edges)

    return {
        "status": "Pipeline Execution Finished",
        "log": execution_log,
        "final_state": pipeline_state,
    }