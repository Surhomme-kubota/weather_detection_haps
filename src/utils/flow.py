

import cv2
import numpy as np

class Flow:
    def __init__(self):
        self.steps = []

    def add_step(self, step_instance):
        """パイプラインに処理ステップを追加する"""
        self.steps.append(step_instance)
        return self

    def execute(self, initial_data=None):
        data = initial_data
        for step in self.steps:
            data = step.process(data)
        return data