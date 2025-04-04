"""HTML visualization utilities for evaluation results."""

import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional, Set
import jinja2
import difflib

def analyze_sentence_alignment(true_sentences: List[str], pred_sentences: List[str]) -> List[Dict[str, Any]]:
    """Analyze and categorize predicted sentences compared to true sentences.
    
    Args:
        true_sentences: List of ground truth sentences
        pred_sentences: List of predicted sentences
        
    Returns:
        List of dicts with sentence text and status (correct, incorrect, missing)
    """
    # First, preprocess the sentences for better comparison
    clean_true_sentences = []
    clean_true_map = {}  # Map from cleaned to original
    
    for i, sent in enumerate(true_sentences):
        if not sent.strip():
            continue
        # Normalize for better matching
        clean_sent = sent.strip().lower()
        clean_true_sentences.append(clean_sent)
        clean_true_map[clean_sent] = sent  # Keep mapping to original
    
    clean_pred_sentences = []
    clean_pred_map = {}  # Map from cleaned to original
    
    for i, sent in enumerate(pred_sentences):
        if not sent.strip():
            continue
        # Normalize for better matching
        clean_sent = sent.strip().lower()
        clean_pred_sentences.append(clean_sent)
        clean_pred_map[clean_sent] = (sent, i)  # Keep mapping to original and index
    
    # Convert to sets for set operations
    true_set = set(clean_true_sentences)
    pred_set = set(clean_pred_sentences)
    
    # Find correct, incorrect and missing sentences
    correct_set = true_set.intersection(pred_set)
    incorrect_set = pred_set - true_set
    missing_set = true_set - pred_set
    
    # Build result list with detailed information
    result = []
    
    # Create a frequency counter for duplicate detection
    pred_freq = {}
    for sent in clean_pred_sentences:
        pred_freq[sent] = pred_freq.get(sent, 0) + 1
    
    # Process each predicted sentence in order
    for clean_sent in clean_pred_sentences:
        original_sent, idx = clean_pred_map[clean_sent]
        
        if clean_sent in correct_set:
            # Correct sentence found in true sentences
            status = 'correct'
            # If this is a duplicate prediction, mark additional instances as incorrect
            if pred_freq[clean_sent] > 1:
                pred_freq[clean_sent] -= 1
                if pred_freq[clean_sent] == 0:
                    # This was the last instance, mark it as correct
                    status = 'correct'
                else:
                    # This is a duplicate, mark as incorrect
                    status = 'incorrect'
        else:
            # Not found in true sentences
            status = 'incorrect'
        
        result.append({
            'text': original_sent,
            'status': status,
            'pred_index': idx
        })
    
    return result

def calculate_example_metrics(true_sentences: List[str], pred_sentences: List[str], 
                             tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate metrics for a single example.
    
    Args:
        true_sentences: List of ground truth sentences
        pred_sentences: List of predicted sentences
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives
        
    Returns:
        Dictionary of metrics
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def process_evaluation_results(results: Dict[str, Any], max_examples_per_dataset: int = 10) -> Dict[str, Any]:
    """Process evaluation results for HTML visualization.
    
    Args:
        results: Evaluation results dictionary
        max_examples_per_dataset: Maximum number of examples to include per dataset
        
    Returns:
        Processed results for template rendering
    """
    processed_data = {
        'title': f"Sentence Tokenizer Evaluation Results ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        'summary_table': [],
        'dataset_details': {},
        'tokenizer_metrics': {},
        'highest_precision': {},
        'highest_recall': {},
        'highest_f1': {},
        'highest_accuracy': {},
        'lowest_time_char': {},
        'lowest_time_sent': {},
        'combined_results': {},
        'highest_combined': {
            'precision': "0.0000",
            'recall': "0.0000",
            'f1': "0.0000",
            'accuracy': "0.0000"
        }
    }
    
    # Initialize combined metrics for each tokenizer
    combined_raw_metrics = {}
    for tokenizer_name in results["tokenizers"]:
        combined_raw_metrics[tokenizer_name] = {
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0,
            'total_sentences': 0,
            'total_characters': 0,
            'total_time_seconds': 0.0
        }
    
    # Process summary metrics for the table
    for tokenizer_name in results["tokenizers"]:
        for dataset_name in results["datasets"]:
            result = results["results"][tokenizer_name][dataset_name]
            summary = result["summary"]
            
            row = [
                tokenizer_name,
                dataset_name,
                f"{summary['precision']:.4f}",
                f"{summary['recall']:.4f}",
                f"{summary['f1']:.4f}",
                f"{summary['accuracy']:.4f}",
                f"{summary['time_per_char_seconds'] * 1000:.4f}",
                f"{summary['time_per_sentence_seconds'] * 1000:.4f}",
                summary['total_sentences'],
                summary['total_chars']
            ]
            processed_data['summary_table'].append(row)
            
            # Accumulate metrics for combined results
            combined_metrics = combined_raw_metrics[tokenizer_name]
            combined_metrics['total_true_positives'] += summary['total_true_positives']
            combined_metrics['total_false_positives'] += summary['total_false_positives']
            combined_metrics['total_false_negatives'] += summary['total_false_negatives']
            combined_metrics['total_sentences'] += summary['total_sentences']
            combined_metrics['total_characters'] += summary['total_chars']
            combined_metrics['total_time_seconds'] += summary.get('total_time_seconds', 0.0)
            
            # Initialize dataset structures if needed
            if dataset_name not in processed_data['dataset_details']:
                processed_data['dataset_details'][dataset_name] = {}
            
            if dataset_name not in processed_data['tokenizer_metrics']:
                processed_data['tokenizer_metrics'][dataset_name] = {}
                # Track both raw values and formatted strings
                processed_data['_raw_metrics'] = processed_data.get('_raw_metrics', {})
                processed_data['_raw_metrics'][dataset_name] = {
                    'highest_precision': 0.0,
                    'highest_recall': 0.0,
                    'highest_f1': 0.0,
                    'highest_accuracy': 0.0,
                    'lowest_time_char': float('inf'),
                    'lowest_time_sent': float('inf')
                }
                # Initialize the display strings
                processed_data['highest_precision'][dataset_name] = "0.0000"
                processed_data['highest_recall'][dataset_name] = "0.0000"
                processed_data['highest_f1'][dataset_name] = "0.0000"
                processed_data['highest_accuracy'][dataset_name] = "0.0000"
                processed_data['lowest_time_char'][dataset_name] = "Inf"
                processed_data['lowest_time_sent'][dataset_name] = "Inf"
            
            # Format for display
            precision_str = f"{summary['precision']:.4f}"
            recall_str = f"{summary['recall']:.4f}"
            f1_str = f"{summary['f1']:.4f}"
            accuracy_str = f"{summary['accuracy']:.4f}"
            time_char_str = f"{summary['time_per_char_seconds'] * 1000:.4f}"
            time_sent_str = f"{summary['time_per_sentence_seconds'] * 1000:.4f}"
            
            # Store metrics for this tokenizer
            processed_data['tokenizer_metrics'][dataset_name][tokenizer_name] = {
                'precision': precision_str,
                'recall': recall_str,
                'f1': f1_str,
                'time_per_char': time_char_str,
                'time_per_sentence': time_sent_str
            }
            
            # Raw values for comparison
            raw_metrics = processed_data['_raw_metrics'][dataset_name]
            
            # Update "best" metrics (compare raw values, store formatted strings)
            if summary['precision'] > raw_metrics['highest_precision']:
                raw_metrics['highest_precision'] = summary['precision']
                processed_data['highest_precision'][dataset_name] = precision_str
                
            if summary['recall'] > raw_metrics['highest_recall']:
                raw_metrics['highest_recall'] = summary['recall']
                processed_data['highest_recall'][dataset_name] = recall_str
                
            if summary['f1'] > raw_metrics['highest_f1']:
                raw_metrics['highest_f1'] = summary['f1']
                processed_data['highest_f1'][dataset_name] = f1_str
                
            if summary['accuracy'] > raw_metrics['highest_accuracy']:
                raw_metrics['highest_accuracy'] = summary['accuracy']
                processed_data['highest_accuracy'][dataset_name] = accuracy_str
                
            if summary['time_per_char_seconds'] * 1000 < raw_metrics['lowest_time_char']:
                raw_metrics['lowest_time_char'] = summary['time_per_char_seconds'] * 1000
                processed_data['lowest_time_char'][dataset_name] = time_char_str
                
            if summary['time_per_sentence_seconds'] * 1000 < raw_metrics['lowest_time_sent']:
                raw_metrics['lowest_time_sent'] = summary['time_per_sentence_seconds'] * 1000
                processed_data['lowest_time_sent'][dataset_name] = time_sent_str
            
            # Process examples for detailed view
            examples_list = []
            
            # First try to get detailed_examples for more comprehensive information
            result_examples = result.get("detailed_examples", [])
            
            # If detailed_examples doesn't exist, fall back to examples
            if not result_examples:
                result_examples = result.get("examples", [])
            
            # Limit the number of examples for performance in the HTML report
            for example in result_examples[:max_examples_per_dataset]:
                true_sentences = example.get("true_sentences", [])
                pred_sentences = example.get("pred_sentences", [])
                
                # Analyze sentence alignment and categorization
                sentence_analysis = analyze_sentence_alignment(true_sentences, pred_sentences)
                
                # Count correct and incorrect sentences based on our analysis
                tp = len([s for s in sentence_analysis if s['status'] == 'correct'])
                fp = len([s for s in sentence_analysis if s['status'] == 'incorrect'])
                # Missing sentences are those in true_sentences but not marked as correct in our analysis
                correct_sentences = set(s.strip().lower() for s in true_sentences if s.strip())
                matched_sentences = set()
                for s in sentence_analysis:
                    if s['status'] == 'correct':
                        matched_sentences.add(s['text'].strip().lower())
                fn = len(correct_sentences) - len(matched_sentences)
                
                # Calculate metrics based on our analysis
                metrics = calculate_example_metrics(true_sentences, pred_sentences, tp, fp, fn)
                
                examples_list.append({
                    'text': example.get("text", ""),
                    'true_sentences': true_sentences,
                    'pred_sentences': pred_sentences,
                    'true_spans': example.get("true_spans", []),
                    'pred_spans': example.get("pred_spans", []),
                    'sentence_analysis': sentence_analysis,
                    'metrics': metrics
                })
            
            processed_data['dataset_details'][dataset_name][tokenizer_name] = examples_list
    
    # Sort the summary table by dataset and then F1 score (descending)
    processed_data['summary_table'].sort(key=lambda x: (x[1], -float(x[4])))
    
    # Calculate combined metrics across all datasets for each tokenizer
    highest_precision = 0.0
    highest_recall = 0.0
    highest_f1 = 0.0
    highest_accuracy = 0.0
    
    for tokenizer_name, raw_metrics in combined_raw_metrics.items():
        tp = raw_metrics['total_true_positives']
        fp = raw_metrics['total_false_positives']
        fn = raw_metrics['total_false_negatives']
        total_sentences = raw_metrics['total_sentences']
        total_chars = raw_metrics['total_characters']
        total_time = raw_metrics['total_time_seconds']
        
        # Calculate combined precision, recall, F1 from raw confusion matrix values
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # Format for display
        precision_str = f"{precision:.4f}"
        recall_str = f"{recall:.4f}"
        f1_str = f"{f1:.4f}"
        accuracy_str = f"{accuracy:.4f}"
        
        # Store combined metrics
        processed_data['combined_results'][tokenizer_name] = {
            'precision': precision_str,
            'recall': recall_str,
            'f1': f1_str,
            'accuracy': accuracy_str,
            'total_sentences': f"{total_sentences:,}",
            'total_characters': f"{total_chars:,}",
            'total_time': f"{total_time:.2f}"
        }
        
        # Track highest metrics
        if precision > highest_precision:
            highest_precision = precision
            processed_data['highest_combined']['precision'] = precision_str
            
        if recall > highest_recall:
            highest_recall = recall
            processed_data['highest_combined']['recall'] = recall_str
            
        if f1 > highest_f1:
            highest_f1 = f1
            processed_data['highest_combined']['f1'] = f1_str
            
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            processed_data['highest_combined']['accuracy'] = accuracy_str
    
    return processed_data

def generate_html_report(results: Dict[str, Any], output_path: str, max_examples_per_dataset: int = 10) -> None:
    """Generate an HTML report of evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the HTML report
        max_examples_per_dataset: Maximum number of examples to include per dataset
    """
    # Create the template environment
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Get the template
    template = env.get_template('evaluation_results.html')
    
    # Process the evaluation results
    processed_data = process_evaluation_results(results, max_examples_per_dataset)
    
    # Find chart images - search in multiple possible chart locations
    output_dir = os.path.dirname(output_path)
    results_dir = os.path.dirname(output_dir)
    
    # Try different potential chart locations
    chart_dirs = [
        os.path.join(results_dir, "charts"),
        os.path.join(results_dir, "publication_charts"),
        os.path.join(results_dir, "publication_charts_final"),
        os.path.join(os.path.dirname(results_dir), "charts")
    ]
    
    chart_paths = {}
    
    # Function to read and encode image as base64
    def encode_image_base64(file_path):
        import base64
        with open(file_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    
    # Search for charts in all potential directories
    for charts_dir in chart_dirs:
        if not os.path.exists(charts_dir):
            continue
            
        # Get comparison charts
        chart_types = ["f1", "precision", "recall", "time-per-char-seconds", "time-per-sentence-seconds"]
        for chart_type in chart_types:
            # Check multiple possible locations and naming patterns
            possible_paths = [
                os.path.join(charts_dir, "comparison", f"{chart_type}.png"),
                os.path.join(charts_dir, f"comparison_{chart_type}.png"),
                os.path.join(charts_dir, f"{chart_type}.png"),
                os.path.join(charts_dir, f"{chart_type}_heatmap.png"),
                os.path.join(charts_dir, f"{chart_type.replace('-', '_')}.png"),
                os.path.join(charts_dir, f"{chart_type.replace('-', '_')}_heatmap.png")
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and chart_type not in chart_paths:
                    chart_paths[chart_type] = encode_image_base64(path)
                    break
        
        # Add tradeoff charts
        tradeoff_dir = os.path.join(charts_dir, "tradeoffs")
        if os.path.exists(tradeoff_dir):
            for chart_file in ["precision_vs_throughput.png", "f1_vs_throughput.png", "recall_vs_throughput.png", "precision_recall.png"]:
                chart_path = os.path.join(tradeoff_dir, chart_file)
                if os.path.exists(chart_path):
                    chart_key = chart_file.replace(".png", "")
                    if f"tradeoff_{chart_key}" not in chart_paths:
                        chart_paths[f"tradeoff_{chart_key}"] = encode_image_base64(chart_path)
        
        # Check for tradeoff charts in main directory as well
        for chart_file in ["precision_vs_time.png", "f1_vs_time.png", "recall_vs_time.png", "precision_recall.png", 
                         "precision_recall_tradeoff.png", "f1_time_tradeoff.png"]:
            chart_path = os.path.join(charts_dir, chart_file)
            if os.path.exists(chart_path):
                chart_key = chart_file.replace(".png", "")
                if f"tradeoff_{chart_key}" not in chart_paths:
                    chart_paths[f"tradeoff_{chart_key}"] = encode_image_base64(chart_path)
        
        # Get weighted metrics chart
        weighted_dir = os.path.join(charts_dir, "weighted")
        if os.path.exists(weighted_dir):
            weighted_chart = os.path.join(weighted_dir, "weighted_metrics.png")
            if os.path.exists(weighted_chart) and "weighted_metrics" not in chart_paths:
                chart_paths["weighted_metrics"] = encode_image_base64(weighted_chart)
                
        # Also check for weighted metrics in main directory
        weighted_chart = os.path.join(charts_dir, "weighted_metrics.png")
        if os.path.exists(weighted_chart) and "weighted_metrics" not in chart_paths:
            chart_paths["weighted_metrics"] = encode_image_base64(weighted_chart)
            
        # Check for dataset comparison chart
        dataset_chart = os.path.join(charts_dir, "dataset_comparison.png")
        if os.path.exists(dataset_chart) and "dataset_comparison" not in chart_paths:
            chart_paths["dataset_comparison"] = encode_image_base64(dataset_chart)
    
    # Add chart paths to the processed data
    processed_data["chart_paths"] = chart_paths
    
    # Render the template
    html_content = template.render(**processed_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_path}")