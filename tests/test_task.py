"""
Unit tests for the Task class
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from taskhlant.task import Task


def test_task_creation():
    """Test creating a new task"""
    task = Task("Buy groceries")
    assert task.title == "Buy groceries"
    assert task.description == ""
    assert not task.completed


def test_task_with_description():
    """Test creating a task with description"""
    task = Task("Buy groceries", "Milk, eggs, bread")
    assert task.title == "Buy groceries"
    assert task.description == "Milk, eggs, bread"
    assert not task.completed


def test_mark_complete():
    """Test marking a task as complete"""
    task = Task("Buy groceries")
    task.mark_complete()
    assert task.completed


def test_mark_incomplete():
    """Test marking a task as incomplete"""
    task = Task("Buy groceries", completed=True)
    task.mark_incomplete()
    assert not task.completed


def test_task_string_representation():
    """Test string representation of tasks"""
    incomplete_task = Task("Incomplete task")
    complete_task = Task("Complete task", completed=True)
    
    assert "[○]" in str(incomplete_task)
    assert "[✓]" in str(complete_task)


if __name__ == "__main__":
    test_task_creation()
    test_task_with_description()
    test_mark_complete()
    test_mark_incomplete()
    test_task_string_representation()
    print("All tests passed!")
