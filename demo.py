#!/usr/bin/env python3
"""
Demo script showing how to use Taskhlant
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from taskhlant.manager import TaskManager


def main():
    """Demo of Taskhlant functionality"""
    print("=" * 50)
    print("Welcome to Taskhlant - Task Management Demo")
    print("=" * 50)
    print()
    
    # Create a task manager
    manager = TaskManager()
    
    # Add some tasks
    print("Adding tasks...")
    task1 = manager.add_task("Buy groceries", "Milk, eggs, bread")
    task2 = manager.add_task("Write documentation")
    task3 = manager.add_task("Review pull requests")
    task4 = manager.add_task("Update dependencies")
    print(f"Added {len(manager)} tasks")
    print()
    
    # Display all tasks
    print("All tasks:")
    for task in manager.get_all_tasks():
        print(f"  {task}")
    print()
    
    # Complete some tasks
    print("Completing some tasks...")
    task1.mark_complete()
    task3.mark_complete()
    print()
    
    # Display completed tasks
    print("Completed tasks:")
    for task in manager.get_completed_tasks():
        print(f"  {task}")
    print()
    
    # Display incomplete tasks
    print("Incomplete tasks:")
    for task in manager.get_incomplete_tasks():
        print(f"  {task}")
    print()
    
    # Summary
    print("=" * 50)
    print(f"Total tasks: {len(manager)}")
    print(f"Completed: {len(manager.get_completed_tasks())}")
    print(f"Remaining: {len(manager.get_incomplete_tasks())}")
    print("=" * 50)


if __name__ == "__main__":
    main()
