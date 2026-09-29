.PHONY: planner precision recall custom-all tools task_complition deepeval-all all

# --- eval_tools/custom (needs: openai, python-dotenv) ---
planner:
	cd eval_tools/custom && python3 custom_metric_agent_plan_test.py --predicted ../../preprocess/extracted_data_tc_3.json --ground-truth ../../input_data/ground_truth_tc_3.json

precision:
	cd eval_tools/custom && python3 custom_metric_agent_plan_precision_test.py --predicted ../../preprocess/extracted_data_tc_3.json --ground-truth ../../input_data/ground_truth_tc_3.json

recall:
	cd eval_tools/custom && python3 custom_metric_agent_plan_recall_test.py --predicted ../../preprocess/extracted_data_tc_3.json --ground-truth ../../input_data/ground_truth_tc_3.json

custom-all: planner precision recall

# --- eval_tools/deepeval (needs: deepeval) ---
tools:
	cd eval_tools/deepeval && python3 tools_call_test.py --predicted ../../preprocess/extracted_data_tc_3.json --ground-truth ../../input_data/ground_truth_tc_3.json

task_complition:
	cd eval_tools/deepeval && python3 task_complition_metric_test.py --history ../../input_data/agent_history_tc_3.json

deepeval-all: tools task_complition

# --- everything ---
all: custom-all deepeval-all