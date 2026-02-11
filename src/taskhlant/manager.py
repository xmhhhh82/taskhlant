"""
Task manager for handling collections of tasks
"""

from .task import Task


class TaskManager:
    """Manages a collection of tasks"""
    
    def __init__(self):
        """Initialize a new task manager"""
        self.tasks = []
    
    def add_task(self, title, description=""):
        """
        Add a new task to the manager
        
        Args:
            title (str): The title of the task
            description (str): Optional description of the task
        
        Returns:
            Task: The newly created task
        """
        task = Task(title, description)
        self.tasks.append(task)
        return task
    
    def remove_task(self, task):
        """
        Remove a task from the manager
        
        Args:
            task (Task): The task to remove
        """
        if task in self.tasks:
            self.tasks.remove(task)
    
    def get_all_tasks(self):
        """
        Get all tasks
        
        Returns:
            list: List of all tasks
        """
        return self.tasks.copy()
    
    def get_completed_tasks(self):
        """
        Get all completed tasks
        
        Returns:
            list: List of completed tasks
        """
        return [task for task in self.tasks if task.completed]
    
    def get_incomplete_tasks(self):
        """
        Get all incomplete tasks
        
        Returns:
            list: List of incomplete tasks
        """
        return [task for task in self.tasks if not task.completed]
    
    def __len__(self):
        """Return the number of tasks"""
        return len(self.tasks)
