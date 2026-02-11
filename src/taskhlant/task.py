"""
Task model for the taskhlant application
"""

class Task:
    """Represents a task in the task management system"""
    
    def __init__(self, title, description="", completed=False):
        """
        Initialize a new task
        
        Args:
            title (str): The title of the task
            description (str): Optional description of the task
            completed (bool): Whether the task is completed
        """
        self.title = title
        self.description = description
        self.completed = completed
    
    def mark_complete(self):
        """Mark the task as completed"""
        self.completed = True
    
    def mark_incomplete(self):
        """Mark the task as incomplete"""
        self.completed = False
    
    def __str__(self):
        """String representation of the task"""
        status = "✓" if self.completed else "○"
        return f"[{status}] {self.title}"
    
    def __repr__(self):
        """Developer-friendly representation of the task"""
        return f"Task(title='{self.title}', completed={self.completed})"
