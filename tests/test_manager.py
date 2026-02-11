"""
Unit tests for the TaskManager class
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from taskhlant.manager import TaskManager


def test_task_manager_creation():
    """Test creating a new task manager"""
    manager = TaskManager()
    assert len(manager) == 0


def test_add_task():
    """Test adding a task to the manager"""
    manager = TaskManager()
    task = manager.add_task("Buy groceries")
    assert len(manager) == 1
    assert task.title == "Buy groceries"


def test_remove_task():
    """Test removing a task from the manager"""
    manager = TaskManager()
    task = manager.add_task("Buy groceries")
    manager.remove_task(task)
    assert len(manager) == 0


def test_get_all_tasks():
    """Test getting all tasks"""
    manager = TaskManager()
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    tasks = manager.get_all_tasks()
    assert len(tasks) == 2


def test_get_completed_tasks():
    """Test getting completed tasks"""
    manager = TaskManager()
    task1 = manager.add_task("Task 1")
    task2 = manager.add_task("Task 2")
    task1.mark_complete()
    
    completed = manager.get_completed_tasks()
    assert len(completed) == 1
    assert task1 in completed


def test_get_incomplete_tasks():
    """Test getting incomplete tasks"""
    manager = TaskManager()
    task1 = manager.add_task("Task 1")
    task2 = manager.add_task("Task 2")
    task1.mark_complete()
    
    incomplete = manager.get_incomplete_tasks()
    assert len(incomplete) == 1
    assert task2 in incomplete


if __name__ == "__main__":
    test_task_manager_creation()
    test_add_task()
    test_remove_task()
    test_get_all_tasks()
    test_get_completed_tasks()
    test_get_incomplete_tasks()
    print("All tests passed!")
