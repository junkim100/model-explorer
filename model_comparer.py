import os
from dotenv import load_dotenv
import fire
import torch
from transformers import AutoModel
import numpy as np
from typing import List, Any
from textual.app import App, ComposeResult
from textual.widgets import Tree, Header, Footer, Static
from textual.containers import Container, ScrollableContainer
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

load_dotenv()

class ModelComparer(App):
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

    #right_container {
        width: 70%;
        height: 100%;
        layout: vertical;
    }

    #details_container {
        width: 100%;
        height: 70%;
        layout: horizontal;
    }

    #details_1, #details_2 {
        width: 50%;
        height: 100%;
    }

    #difference_container {
        width: 100%;
        height: 30%;
    }
    """


    def __init__(self, model_name_1: str, model_name_2: str):
        super().__init__()
        self.model_name_1 = model_name_1
        self.model_name_2 = model_name_2
        self.model_1 = None
        self.model_2 = None
        self.tensor_offset = {}
        self.selected_tensors = [None, None]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="model_container"):
            yield Tree("Models", id="model_tree")
            with Container(id="right_container"):
                with Container(id="details_container"):
                    yield Static(id="details_1")
                    yield Static(id="details_2")
                yield Static(id="difference_container")
        yield Footer()


    def on_mount(self) -> None:
        try:
            self.model_1 = AutoModel.from_pretrained(self.model_name_1)
            self.model_2 = AutoModel.from_pretrained(self.model_name_2)
            self.populate_tree()
            self.show_model_summary()
        except Exception as e:
            self.show_error(f"Error loading models: {str(e)}")

    def show_error(self, message: str):
        details_1 = self.query_one("#details_1")
        details_2 = self.query_one("#details_2")
        content = Text(message, style="bold red")
        panel = Panel(content, title="Error", border_style="red")
        details_1.update(panel)
        details_2.update(panel)

    def show_model_summary(self):
        details_1 = self.query_one("#details_1")
        details_2 = self.query_one("#details_2")

        for i, (model_name, model) in enumerate([(self.model_name_1, self.model_1), (self.model_name_2, self.model_2)]):
            if model is None:
                content = Text("Model not loaded. Please check for errors.", style="bold red")
                panel = Panel(content, title="Model Summary", border_style="red")
            else:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                summary = f"""
                Model: {model_name}
                Total parameters: {total_params:,}
                Trainable parameters: {trainable_params:,}
                Layers: {len(list(model.modules()))}
                """
                content = Text(summary.strip())
                panel = Panel(content, title="Model Summary", border_style="green")

            if i == 0:
                details_1.update(panel)
            else:
                details_2.update(panel)

    def populate_tree(self):
        root = self.query_one("#model_tree").root
        model_1_node = root.add(self.model_name_1)
        model_2_node = root.add(self.model_name_2)
        self.populate_model_tree(model_1_node, self.model_1, "model_1")
        self.populate_model_tree(model_2_node, self.model_2, "model_2")

    def populate_model_tree(self, node, current_layer, path):
        if isinstance(current_layer, torch.nn.ModuleList):
            for i, layer in enumerate(current_layer):
                child_node = node.add(f"Layer {i+1}")
                self.populate_model_tree(child_node, layer, f"{path}.[{i}]")
        elif isinstance(current_layer, torch.nn.Module):
            for name, child in current_layer.named_children():
                child_node = node.add(name)
                if name == "rotary_emb":
                    child_node.allow_expand = False
                else:
                    self.populate_model_tree(child_node, child, f"{path}.{name}")
            for name, param in current_layer.named_parameters(recurse=False):
                param_node = node.add(f"{name}")
                if param.dim() > 0:
                    self.tensor_offset[f"{path}.{name}"] = 0
                    self.add_tensor_elements(param_node, param, f"{path}.{name}")
                else:
                    param_node.allow_expand = False

    def add_tensor_elements(self, node, tensor, path):
        if path not in self.tensor_offset:
            self.tensor_offset[path] = 0
        offset = self.tensor_offset[path]
        for i in range(offset, min(offset + 20, tensor.size(0))):
            element_node = node.add(f"Element {i+1}")
            element_node.allow_expand = False
        if tensor.size(0) > offset + 20:
            next_node = node.add("... (Next)")
            next_node.allow_expand = False
        if offset > 0:
            prev_node = node.add("... (Previous)")
            prev_node.allow_expand = False

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        node = event.node
        path = self.get_node_path(node)
        if not path:
            self.show_model_summary()
            return
        if path[-1] in ["... (Next)", "... (Previous)"]:
            return

        model_name = path[0]
        model = self.model_1 if model_name == self.model_name_1 else self.model_2
        current_layer = model
        for part in path[1:]:
            part_str = str(part)
            if part_str.startswith("Layer "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            elif part_str.startswith("Element "):
                index = int(part_str.split()[1]) - 1
                current_layer = current_layer[index]
            else:
                current_layer = getattr(current_layer, part_str)

        self.update_details(current_layer, str(path[-1]), model_name)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        path = self.get_node_path(node)
        if not path:
            self.show_model_summary()
            return
        if str(path[-1]) in ["... (Next)", "... (Previous)"]:
            self.navigate_tensor(node)
        else:
            model_name = path[0]
            model = self.model_1 if model_name == self.model_name_1 else self.model_2
            current_layer = model
            for part in path[1:]:
                part_str = str(part)
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

            self.update_details(current_layer, str(path[-1]), model_name)
            self.update_selected_tensor(current_layer, model_name)

    def get_node_path(self, node):
        path = []
        while node.parent is not None:
            path.insert(0, node.label.plain)
            node = node.parent
        return path

    def update_details(self, current_layer, part, model_name):
        details = self.query_one("#details_1" if model_name == self.model_name_1 else "#details_2")
        content = Text()

        if isinstance(current_layer, torch.Tensor):
            if current_layer.dim() == 0:
                content.append(f"Scalar Value: {current_layer.item()}")
            else:
                content.append(f"Tensor Shape: {current_layer.shape}\n")
                content.append(f"Tensor Type: {current_layer.dtype}\n")
                content.append(f"Requires Grad: {current_layer.requires_grad}\n")
                if part.startswith("Element "):
                    content.append("Tensor Values (First 20):\n")
                    content.append(str(current_layer.flatten()[:min(20, current_layer.numel())].tolist()))
        elif isinstance(current_layer, torch.nn.Module):
            params = sum(p.numel() for p in current_layer.parameters())
            content.append(f"Module: {type(current_layer).__name__}\n")
            content.append(f"Parameters: {params:,}\n")
            for name, param in current_layer.named_parameters():
                content.append(f"\n{name}:\n")
                content.append(f" Shape: {param.shape}\n")
                content.append(f" Type: {param.dtype}\n")
                content.append(f" Requires Grad: {param.requires_grad}\n")
        elif isinstance(current_layer, torch.nn.Parameter):
            content.append(f"Parameter Shape: {current_layer.shape}\n")
            content.append(f"Parameter Type: {current_layer.dtype}\n")
            content.append(f"Requires Grad: {current_layer.requires_grad}\n")
            content.append("Parameter Values:\n")
            content.append(str(current_layer.flatten()[:min(20, current_layer.numel())].tolist()))
        else:
            content.append(f"Type: {type(current_layer).__name__}")

        panel = Panel(content, title=f"{model_name}: {str(part)}", border_style="blue")
        details.update(panel)

    def update_selected_tensor(self, tensor, model_name):
        index = 0 if model_name == self.model_name_1 else 1
        if isinstance(tensor, torch.Tensor):
            self.selected_tensors[index] = tensor
        else:
            self.selected_tensors[index] = None
        self.update_difference_view()

    def update_difference_view(self):
        difference_container = self.query_one("#difference_container")
        
        if self.selected_tensors[0] is None or self.selected_tensors[1] is None:
            difference_container.update("Select tensor elements from both models to compare")
            return

        if not isinstance(self.selected_tensors[0], torch.Tensor) or not isinstance(self.selected_tensors[1], torch.Tensor):
            difference_container.update("Selected items are not comparable tensors")
            return

        if self.selected_tensors[0].shape != self.selected_tensors[1].shape:
            difference_container.update("Selected tensors have different shapes")
            return

        difference = (self.selected_tensors[0] - self.selected_tensors[1]).abs()
        flat_difference = difference.flatten()
        top_10_values, top_10_indices = torch.topk(flat_difference, min(10, flat_difference.numel()))

        table = Table(title="Top 10 Differences", show_header=True, header_style="bold magenta")
        table.add_column("Location", style="dim", width=20)
        table.add_column("Difference", justify="right")
        table.add_column(f"Value in {self.model_name_1}", justify="right")
        table.add_column(f"Value in {self.model_name_2}", justify="right")

        for value, index in zip(top_10_values, top_10_indices):
            location = np.unravel_index(index.item(), difference.shape)
            value_1 = self.selected_tensors[0][location].item()
            value_2 = self.selected_tensors[1][location].item()
            table.add_row(
                str(location),
                f"{value.item():.6f}",
                f"{value_1:.6f}",
                f"{value_2:.6f}"
            )

        panel = Panel(table, title="Tensor Difference", border_style="green")
        difference_container.update(panel)

    def navigate_tensor(self, node):
        path = self.get_node_path(node)
        model_name = path[0]
        param_path = ".".join(path[1:-1])
        param = self.get_param_by_path(param_path, model_name)
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

    def get_param_by_path(self, path, model_name):
        current = self.model_1 if model_name == self.model_name_1 else self.model_2
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

def compare_models(model_name_1: str, model_name_2: str):
    app = ModelComparer(model_name_1, model_name_2)
    app.run()

if __name__ == "__main__":
    fire.Fire(compare_models)
