# Taskhlant Documentation

## Overview

Taskhlant is a simple task management application written in Python.

## Architecture

The project is structured as follows:

- `src/taskhlant/` - Main application code
  - `task.py` - Task model
  - `manager.py` - Task manager for handling collections of tasks
- `tests/` - Unit tests
- `docs/` - Documentation

## Usage Examples

### Creating a Task

```python
from taskhlant.task import Task

task = Task("Buy groceries", "Milk, eggs, bread")
print(task)  # [○] Buy groceries
```

### Managing Tasks

```python
from taskhlant.manager import TaskManager

manager = TaskManager()
task1 = manager.add_task("Task 1")
task2 = manager.add_task("Task 2")

task1.mark_complete()

print(f"Total tasks: {len(manager)}")
print(f"Completed: {len(manager.get_completed_tasks())}")
print(f"Incomplete: {len(manager.get_incomplete_tasks())}")
```

## Testing

Run the tests using:

```bash
python tests/test_task.py
python tests/test_manager.py
```

## Future Enhancements

- Persistent storage (JSON/SQLite)
- Task priorities
- Due dates
- Categories/tags
- CLI interface
- Web interface
