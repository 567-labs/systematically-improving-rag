This directory shows how to test whether we are retrieving the correct tools. It includes

- `benchmark_tool_retrieval.ipynb`: Notebook showing the workflow
- `utils.py`: Reusable/modifiable utility functions to calculate recall & precision of tools
- `funcs_to_call.py`: Mocked objects showing the list of available functions. They are not implemented here, because we would not be testing the implementation anyway. We are just testing whether we can retrieve the correct tools.
