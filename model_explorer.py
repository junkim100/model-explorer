import os
from dotenv import load_dotenv
import fire
import torch
from transformers import AutoModel
from typing import List, Any

load_dotenv()


class ModelExplorer:
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        self.current_path: List[str] = []

    def display_breadcrumbs(self):
        print(" > ".join(["Model"] + self.current_path))

    def display_options(self, current_layer: Any):
        if isinstance(current_layer, torch.nn.ModuleList):
            print("Options:")
            for i, _ in enumerate(current_layer):
                print(f"  {i}")
        elif isinstance(current_layer, torch.nn.Module):
            print("Options:")
            for name, child in current_layer.named_children():
                print(f"  {name}")
            for name, param in current_layer.named_parameters(recurse=False):
                print(f"  {name} (parameter)")
        elif isinstance(current_layer, torch.Tensor):
            print(f"Tensor Shape: {current_layer.shape}")
            print(f"Tensor Type: {current_layer.dtype}")
            print(f"Tensor Values (first 10): {current_layer.flatten()[:10]}")
        else:
            print("Options: None")

    def navigate(self, path: str = ""):
        if path:
            self.current_path = path.split(".")

        current_layer = self.model
        for part in self.current_path:
            if part.isdigit() and isinstance(current_layer, torch.nn.ModuleList):
                current_layer = current_layer[int(part)]
            elif hasattr(current_layer, part):
                current_layer = getattr(current_layer, part)
            else:
                print(f"Invalid path: {part} not found")
                return

        self.display_breadcrumbs()
        self.display_options(current_layer)

        if isinstance(current_layer, torch.Tensor):
            input("Press Enter to continue...")
            self.current_path.pop()
            self.navigate()
            return

        choice = input("Enter your choice (or 'back', 'exit'): ")
        if choice == "back":
            if self.current_path:
                self.current_path.pop()
            self.navigate()
        elif choice == "exit":
            return
        else:
            self.current_path.append(choice)
            self.navigate()


def explore_model(model_name: str):
    explorer = ModelExplorer(model_name)
    explorer.navigate()


if __name__ == "__main__":
    fire.Fire(explore_model)
