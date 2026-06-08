import unittest
from generated_program import TaskManager

class Tests(unittest.TestCase):
    def test_task_manager_starts_empty(self):
        manager = TaskManager()
        self.assertEqual(len(manager.list_tasks()), 0)

    def test_add_task_with_title_and_priority(self):
        manager = TaskManager()
        task1 = manager.add_task("Write tests", priority="high")
        self.assertIsNotNone(task1.id)
        self.assertEqual(task1.title, "Write tests")
        self.assertEqual(task1.priority, "high")

    def test_add_task_with_default_priority(self):
        manager = TaskManager()
        task2 = manager.add_task("Implement code")
        self.assertIsNotNone(task2.id)
        self.assertEqual(task2.title, "Implement code")
        self.assertEqual(task2.priority, "medium")  # Assuming default priority is medium

    def test_add_task_with_invalid_title(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.add_task(" ", priority="high")

    def test_add_task_with_invalid_priority(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.add_task("Clean up", priority="urgent")

    def test_get_existing_task_by_id(self):
        manager = TaskManager()
        task3 = manager.add_task("Clean up", priority="low")
        retrieved_task = manager.get_task(task3.id)
        self.assertEqual(retrieved_task.title, "Clean up")

    def test_get_missing_task_by_id(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.get_task(12345)

    def test_complete_existing_task(self):
        manager = TaskManager()
        task4 = manager.add_task("Test completion")
        manager.complete_task(task4.id)
        retrieved_task = manager.get_task(task4.id)
        self.assertTrue(retrieved_task.completed)

    def test_complete_already_completed_task(self):
        manager = TaskManager()
        task5 = manager.add_task("Complete already", priority="low")
        manager.complete_task(task5.id)
        with self.assertRaises(ValueError):
            manager.complete_task(task5.id)

    def test_complete_missing_task(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.complete_task(12345)

    def test_delete_existing_task(self):
        manager = TaskManager()
        task6 = manager.add_task("Delete me")
        manager.delete_task(task6.id)
        with self.assertRaises(ValueError):
            manager.get_task(task6.id)

    def test_delete_missing_task(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.delete_task(12345)

    def test_list_all_tasks(self):
        manager = TaskManager()
        task7 = manager.add_task("Task 1")
        task8 = manager.add_task("Task 2", priority="high")
        all_tasks = manager.list_tasks()
        self.assertEqual(len(all_tasks), 2)

    def test_list_completed_tasks(self):
        manager = TaskManager()
        task9 = manager.add_task("Completed task")
        manager.complete_task(task9.id)
        completed_tasks = manager.list_tasks(completed=True)
        self.assertEqual(len(completed_tasks), 1)

    def test_list_incomplete_tasks(self):
        manager = TaskManager()
        task10 = manager.add_task("Incomplete task")
        incomplete_tasks = manager.list_tasks(completed=False)
        self.assertEqual(len(incomplete_tasks), 1)

    def test_list_tasks_by_priority(self):
        manager = TaskManager()
        task11 = manager.add_task("High priority", priority="high")
        high_priority_tasks = manager.list_tasks(priority="high")
        self.assertEqual(len(high_priority_tasks), 1)

    def test_list_tasks_with_combined_filters(self):
        manager = TaskManager()
        task12 = manager.add_task("Completed and high", priority="high")
        manager.complete_task(task12.id)
        completed_high_priority_tasks = manager.list_tasks(completed=True, priority="high")
        self.assertEqual(len(completed_high_priority_tasks), 1)

    def test_list_missing_filter_value(self):
        manager = TaskManager()
        with self.assertRaises(ValueError):
            manager.list_tasks(priority="urgent")

if __name__ == '__main__':
    unittest.main()
