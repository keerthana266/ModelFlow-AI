# app/ml/transpiler.py
import pandas as pd
from typing import Dict, Any

def execute_pipeline_graph(nodes: list, edges: list) -> dict:
    """
    Parses a React Flow JSON graph and executes the ML steps in the correct order.
    """
    
    # 1. Map nodes by ID for quick lookup
    node_map = {node['id']: node for node in nodes}
    
    # 2. Find the root node (A node with no incoming edges)
    # For Sprint 1, we assume a simple linear flow: DataLoader -> Model
    target_nodes = set(edge['target'] for edge in edges)
    root_nodes = [n['id'] for n in nodes if n['id'] not in target_nodes]
    
    if not root_nodes:
        raise ValueError("Invalid pipeline: No starting node found.")
    
    current_node_id = root_nodes[0]
    pipeline_state: Dict[str, Any] = {} # This holds data as it passes between nodes
    execution_log = []

    # 3. Traverse the graph and execute logic based on the node 'type'
    while current_node_id:
        node = node_map[current_node_id]
        node_type = node['type']
        node_data = node.get('data', {})
        
        execution_log.append(f"Executing node: {node_type} ({current_node_id})")

        try:
            if node_type == "dataloader":
                # Mock logic: Load the dataset path provided in the node
                dataset_id = node_data.get("dataset_id")
                target_col = node_data.get("target_column")
                
                pipeline_state['dataset_id'] = dataset_id
                pipeline_state['target_column'] = target_col
                execution_log.append(f"-> Loaded config for dataset {dataset_id}")

            elif node_type == "automl_model":
                # Mock logic: Trigger the AutoML training using data from previous nodes
                if 'dataset_id' not in pipeline_state:
                    raise ValueError("Cannot train model: No dataset loaded in pipeline.")
                
                # In a real scenario, we would trigger the Celery task here.
                # For this transpiler test, we just log the intent.
                execution_log.append(f"-> Triggering AutoML for target: {pipeline_state.get('target_column')}")
                pipeline_state['model_status'] = "Training Initiated"

            else:
                execution_log.append(f"-> Warning: Unknown node type '{node_type}' ignored.")

        except Exception as e:
            execution_log.append(f"-> ERROR at node {current_node_id}: {str(e)}")
            break # Stop pipeline on error

        # 4. Find the next node in the graph
        next_edges = [edge for edge in edges if edge['source'] == current_node_id]
        current_node_id = next_edges[0]['target'] if next_edges else None

    return {
        "status": "Pipeline Execution Finished",
        "log": execution_log,
        "final_state": pipeline_state
    }