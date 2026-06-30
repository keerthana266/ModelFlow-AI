# app/ml/transpiler.py

from typing import Any, Dict, List, Optional

from app.ml.trainer import train_automl
from app.services.storage import get_dataset_path


def normalize_node_type(node_type: str) -> str:
    return node_type.replace("_", "").replace("-", "").lower()


def validate_graph(nodes: List[dict], edges: List[dict]) -> None:
    if not nodes:
        raise ValueError("Pipeline must contain at least one node.")

    node_ids = {node.get("id") for node in nodes}

    if None in node_ids:
        raise ValueError("Every pipeline node must have an id.")

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in node_ids:
            raise ValueError(f"Invalid edge: source node '{source}' does not exist.")

        if target not in node_ids:
            raise ValueError(f"Invalid edge: target node '{target}' does not exist.")


def find_root_node_id(nodes: List[dict], edges: List[dict]) -> str:
    target_nodes = {edge["target"] for edge in edges}
    root_nodes = [node["id"] for node in nodes if node["id"] not in target_nodes]

    if not root_nodes:
        raise ValueError("Invalid pipeline: No starting node found. Possible cycle detected.")

    if len(root_nodes) > 1:
        raise ValueError(
            "Invalid pipeline: Multiple starting nodes found. "
            "For now, only one linear pipeline is supported."
        )

    return root_nodes[0]


def find_next_node_id(current_node_id: str, edges: List[dict]) -> Optional[str]:
    outgoing_edges = [edge for edge in edges if edge["source"] == current_node_id]

    if not outgoing_edges:
        return None

    if len(outgoing_edges) > 1:
        raise ValueError(
            f"Node '{current_node_id}' has multiple outgoing edges. "
            "Branching pipelines are not supported yet."
        )

    return outgoing_edges[0]["target"]


def parse_time_budget(value: Any) -> int:
    try:
        time_budget = int(value)
    except (TypeError, ValueError):
        time_budget = 30

    return max(10, min(time_budget, 300))


def execute_pipeline_graph(nodes: List[dict], edges: List[dict]) -> dict:
    validate_graph(nodes, edges)

    node_map = {node["id"]: node for node in nodes}
    current_node_id = find_root_node_id(nodes, edges)

    pipeline_state: Dict[str, Any] = {}
    execution_log: List[str] = []
    visited_nodes = set()

    while current_node_id:
        if current_node_id in visited_nodes:
            raise ValueError("Invalid pipeline: Cycle detected in graph.")

        visited_nodes.add(current_node_id)

        node = node_map[current_node_id]
        raw_node_type = node.get("type", "")
        node_type = normalize_node_type(raw_node_type)
        node_data = node.get("data", {}) or {}

        execution_log.append(f"Executing node: {raw_node_type} ({current_node_id})")

        try:
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
                execution_log.append(f"-> Target column selected: {target_column}")

            elif node_type == "automl":
                dataset_id = pipeline_state.get("dataset_id")
                target_column = pipeline_state.get("target_column")

                if not dataset_id or not target_column:
                    raise ValueError(
                        "Cannot start AutoML: No dataset configuration found. "
                        "Please connect a Data Loader node before AutoML."
                    )

                time_budget = parse_time_budget(node_data.get("time_budget", 30))
                dataset_path = get_dataset_path(dataset_id)

                execution_log.append(f"-> Triggering AutoML training for target: {target_column}")
                execution_log.append(f"-> Time budget: {time_budget} seconds")

                task = train_automl.delay(
                    dataset_path_str=str(dataset_path),
                    target_column=target_column,
                    time_budget=time_budget,
                )

                pipeline_state["task_id"] = task.id
                pipeline_state["model_status"] = "Training Initiated"
                pipeline_state["time_budget"] = time_budget

                execution_log.append(f"-> Background Celery task queued: {task.id}")

            else:
                execution_log.append(f"-> Warning: Unknown node type '{raw_node_type}' ignored.")

        except Exception as error:
            execution_log.append(f"-> ERROR at node {current_node_id}: {str(error)}")
            raise error

        current_node_id = find_next_node_id(current_node_id, edges)

    return {
        "status": "Pipeline Execution Finished",
        "log": execution_log,
        "final_state": pipeline_state,
    }