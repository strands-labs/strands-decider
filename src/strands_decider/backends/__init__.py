"""Runtimes for model checkpoints, one framework each, imported on their own so that a runtime's
dependencies are needed only where it runs. transformers.py implements
`strands_decider.decision.DecisionBackend`; llama_cpp.py, without torch, implements
`strands_decider.core.Backend` for the Lambda function."""
