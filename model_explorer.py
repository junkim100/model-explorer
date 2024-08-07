import os
from dotenv import load_dotenv
import fire
import torch
from transformers import AutoModel
from typing import List, Any
from textual.app import App, ComposeResult
from textual.widgets import Tree, Header, Footer, Static
from textual.containers import Container, ScrollableContainer
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel

load_dotenv()


class ModelExplorer(App):
    BINDINGS = [Binding("q", "quit", "Quit")]

    CSS = """
    #model_container {
        layout: horizontal;
        height: 100%;
    }
    #model_tree {
        width: 30%;
        height: 100%;
        overflow: auto;
    }
    #details_container {
        width: 70%;
        height: 100%;
    }
    #details {
        width: 100%;
    }
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tensor_offset = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="model_container"):
            yield Tree(self.model_name, id="model_tree")
            with ScrollableContainer(id="details_container"):
                yield Static(id="details")
        yield Footer()

    def on_mount(self) -> None:
        self.populate_tree()

    def populate_tree(self, node=None, current_layer=None, path=""):
        if node is None:
            node = self.query_one("#model_tree").root
            current_layer = self.model

        if isinstance(current_layer, torch.nn.ModuleList):
            for i, layer in enumerate(current_layer):
                child_node = node.add(f"Layer {i+1}")
                self.populate_tree(child_node, layer, f"{path}.[{i}]")
        elif isinstance(current_layer, torch.nn.Module):
            for name, child in current_layer.named_children():
                child_node = node.add(name)
                if name == "rotary_emb":
                    child_node.allow_expand = (
                        False  # Remove collapse icon for rotary_emb
                    )
                else:
                    self.populate_tree(child_node, child, f"{path}.{name}")
            for name, param in current_layer.named_parameters(recurse=False):
                param_node = node.add(f"{name}")
                if param.dim() > 0:
                    self.tensor_offset[f"{path}.{name}"] = 0
                    self.add_tensor_elements(param_node, param, f"{path}.{name}")
        else:
            # This is a leaf node (e.g., a single tensor or parameter)
            node.allow_expand = False  # Remove collapse icon for leaf nodes

    def add_tensor_elements(self, node, tensor, path):
        if path not in self.tensor_offset:
            self.tensor_offset[path] = 0
        offset = self.tensor_offset[path]
        for i in range(offset, min(offset + 20, tensor.size(0))):
            element_node = node.add(f"Element {i+1}")
            element_node.allow_expand = (
                False  # Remove collapse icon for tensor elements
            )
        if tensor.size(0) > offset + 20:
            next_node = node.add("... (Next)")
            next_node.allow_expand = False
        if offset > 0:
            prev_node = node.add("... (Previous)")
            prev_node.allow_expand = False

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        node = event.node
        path = self.get_node_path(node)

        if not path:  # Handle the case when the root node is highlighted
            self.update_details(self.model, "Model")
            return

        if path[-1] in ["... (Next)", "... (Previous)"]:
            # Don't update details for navigation nodes
            return

        current_layer = self.model
        for part in path:
            part_str = str(part)
            if part_str.startswith("Layer "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            elif part_str.startswith("Element "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            else:
                current_layer = getattr(current_layer, part_str)

        self.update_details(current_layer, str(path[-1]))

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        path = self.get_node_path(node)

        if not path:  # Handle the case when the root node is selected
            self.update_details(self.model, "Model")
            return

        if str(path[-1]) in ["... (Next)", "... (Previous)"]:
            self.navigate_tensor(node)
        else:
            current_layer = self.model
            for part in path:
                part_str = str(part)  # Convert Text to string
                if part_str.startswith("Layer "):
                    index = int(part_str.split()[1]) - 1
                    current_layer = current_layer[index]
                elif part_str.startswith("Element "):
                    index = int(part_str.split()[1]) - 1
                    if isinstance(current_layer, torch.nn.Parameter):
                        current_layer = current_layer.data[index]
                    else:
                        current_layer = current_layer[index]
                else:
                    current_layer = getattr(current_layer, part_str)

            self.update_details(current_layer, str(path[-1]))

    def get_node_path(self, node):
        path = []
        while node.parent is not None:
            path.insert(0, node.label.plain)  # Use .plain instead of str()
            node = node.parent
        return path

    def update_details_for_path(self, path):
        current_layer = self.model
        for part in path:
            part_str = str(part)
            if part_str.startswith("Layer "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            elif part_str.startswith("Element "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            else:
                current_layer = getattr(current_layer, part_str)

        self.update_details(current_layer, path[-1])

    def update_details(self, current_layer, part):
        details = self.query_one("#details")
        content = Text()

        if isinstance(current_layer, torch.Tensor):
            if current_layer.dim() == 0:
                content.append(f"Scalar Value: {current_layer.item()}")
            else:
                content.append(f"Tensor Shape: {current_layer.shape}\n")
                content.append(f"Tensor Type: {current_layer.dtype}\n")
                content.append(f"Requires Grad: {current_layer.requires_grad}\n")
                content.append("Tensor Values:\n")
                content.append(
                    str(current_layer.flatten()[: current_layer.shape[0]].tolist())
                )
        elif isinstance(current_layer, torch.nn.Module):
            params = sum(p.numel() for p in current_layer.parameters())
            content.append(f"Module: {type(current_layer).__name__}\n")
            content.append(f"Parameters: {params}\n")
            for name, param in current_layer.named_parameters():
                content.append(f"\n{name}:\n")
                content.append(f"  Shape: {param.shape}\n")
                content.append(f"  Type: {param.dtype}\n")
                content.append(f"  Requires Grad: {param.requires_grad}\n")
        elif isinstance(current_layer, torch.nn.Parameter):
            content.append(f"Parameter Shape: {current_layer.shape}\n")
            content.append(f"Parameter Type: {current_layer.dtype}\n")
            content.append(f"Requires Grad: {current_layer.requires_grad}\n")
            content.append("Parameter Values:\n")
            content.append(
                str(current_layer.flatten()[: current_layer.shape[0]].tolist())
            )
        else:
            content.append(f"Type: {type(current_layer).__name__}")

        panel = Panel(content, title=str(part), border_style="blue")
        details.update(panel)

    def navigate_tensor(self, node):
        path = self.get_node_path(node)

        param_path = ".".join(path[:-1])
        param = self.get_param_by_path(param_path)
        if param_path not in self.tensor_offset:
            self.tensor_offset[param_path] = 0
        if path[-1] == "... (Next)":
            self.tensor_offset[param_path] += 20
        else:
            self.tensor_offset[param_path] = max(0, self.tensor_offset[param_path] - 20)
        parent_node = node.parent
        if parent_node is not None:
            parent_node.remove_children()
            self.add_tensor_elements(parent_node, param, param_path)

    def get_param_by_path(self, path):
        current = self.model
        for part in path.split("."):
            if part.startswith("Layer "):
                index = int(part.split()[1]) - 1
                current = current[index]
            elif part.startswith("Element "):
                index = int(part.split()[1]) - 1
                if isinstance(current, torch.nn.Parameter):
                    current = current.data[index]
                else:
                    current = current[index]
            else:
                current = getattr(current, part)
        return current

    def action_quit(self):
        self.exit()


def explore_model(model_name: str):
    app = ModelExplorer(model_name)
    app.run()


if __name__ == "__main__":
    fire.Fire(explore_model)
