from typing import Dict, Any
from app.ml.trainer import train_automl
from app.services.storage import get_dataset_path

def execute_pipeline_graph(nodes: list, edges: list) -> dict:
    """
    Parses a React Flow JSON graph and executes the ML steps in the correct order.
    """
    node_map = {node['id']: node for node in nodes}
    
    target_nodes = set(edge['target'] for edge in edges)
    root_nodes = [n['id'] for n in nodes if n['id'] not in target_nodes]
    
    if not root_nodes:
        raise ValueError("Invalid pipeline: No starting node found.")
    
    current_node_id = root_nodes[0]
    pipeline_state: Dict[str, Any] = {} 
    execution_log = []

    while current_node_id:
        node = node_map[current_node_id]
        node_type = node['type'].lower()
        node_data = node.get('data', {})
        
        execution_log.append(f"Executing node: {node_type} ({current_node_id})")

        try:
            if node_type == "dataloader":
                dataset_id = node_data.get("dataset_id")
                target_col = node_data.get("target_column")
                
                if not dataset_id or not target_col:
                    raise ValueError("Data Loader node is missing a Dataset ID or Target Column.")
                
                pipeline_state['dataset_id'] = dataset_id
                pipeline_state['target_column'] = target_col
                execution_log.append(f"-> Loaded config for dataset {dataset_id}")

            elif node_type == "automl":
                # Ensure the previous node passed down the dataset info
                dataset_id = pipeline_state.get('dataset_id')
                target_col = pipeline_state.get('target_column')
                
                if not dataset_id or not target_col:
                    raise ValueError("Cannot train model: No dataset loaded in pipeline.")
                
                execution_log.append(f"-> Triggering AutoML for target: {target_col}")
                
                # REAL LOGIC: Get the file path and trigger the Celery background task
                dataset_path = get_dataset_path(dataset_id)
                task = train_automl.delay(
                    dataset_path_str=str(dataset_path),
                    target_column=target_col
                )
                
                pipeline_state['task_id'] = task.id
                pipeline_state['model_status'] = "Training Initiated"
                execution_log.append(f"-> Background Task queued with ID: {task.id}")

            else:
                execution_log.append(f"-> Warning: Unknown node type '{node_type}' ignored.")

        except Exception as e:
            execution_log.append(f"-> ERROR at node {current_node_id}: {str(e)}")
            raise e

        # Find the next node in the graph
        next_edges = [edge for edge in edges if edge['source'] == current_node_id]
        current_node_id = next_edges[0]['target'] if next_edges else None

    return {
        "status": "Pipeline Execution Finished",
        "log": execution_log,
        "final_state": pipeline_state
    }