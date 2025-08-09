import lark
import pandas as pd
from ml import load_dataset, train_model, predict_model, save_model, load_model
from utils import inkl_print
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

grammar = """
start: statement*
statement: print_stmt
         | dataset_stmt
         | train_stmt
         | predict_stmt
         | savemodel_stmt
         | loadmodel_stmt
         | if_stmt
         | repeat_stmt

print_stmt: "print" STRING
dataset_stmt: "dataset" NAME "from" STRING
train_stmt: "acc" "=" "train" NAME "with" NAME ("using" NAME)? "for" NUMBER "epochs"
predict_stmt: "predict" NAME "with" NAME
savemodel_stmt: "savemodel" NAME "to" STRING
loadmodel_stmt: "loadmodel" NAME "from" STRING
if_stmt: "if" condition ":" statement+
repeat_stmt: "repeat" NUMBER "times" ":" statement+

condition: ACC CMP_OP NUMBER
ACC: "acc"
CMP_OP: ">" | "<" | "==" | ">=" | "<=" | "!="
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
STRING: /".*?"/
NUMBER: /[0-9]*\.[0-9]+|[0-9]+/
COMMENT: /#.*?$/m
%ignore COMMENT
%ignore /\s+/
"""

class InklangRuntime:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.variables = {}
    
    def run_statements(self, statements):
        logging.debug(f"Running statements: {statements}")
        for stmt in statements:
            try:
                if stmt[0] == 'print':
                    inkl_print(stmt[1].strip('"'))
                elif stmt[0] == 'dataset':
                    dataset_name, path = stmt[1], stmt[2].strip('"')
                    self.datasets[dataset_name] = load_dataset(dataset_name, path)
                elif stmt[0] == 'train':
                    model_name, dataset_name, model_type, epochs = stmt[1], stmt[2], stmt[3] or 'NeuralNetwork', int(stmt[4])
                    dataset = self.datasets.get(dataset_name)
                    if dataset is None:
                        dataset = load_dataset(dataset_name)
                    if dataset is None:
                        raise ValueError(f"[inklang] Dataset '{dataset_name}' not found")
                    model, metric = train_model(model_name, dataset, model_type, epochs)
                    if model is None:
                        raise ValueError(f"[inklang] Failed to train model '{model_name}'")
                    self.models[model_name] = model
                    self.variables['acc'] = metric
                elif stmt[0] == 'predict':
                    model_name, dataset_name = stmt[1], stmt[2]
                    model = self.models.get(model_name)
                    dataset = self.datasets.get(dataset_name)
                    if dataset is None:
                        dataset = load_dataset(dataset_name)
                    if model is None:
                        raise ValueError(f"[inklang] Model '{model_name}' not found")
                    if dataset is None:
                        raise ValueError(f"[inklang] Dataset '{dataset_name}' not found")
                    predict_model(model, dataset)
                elif stmt[0] == 'savemodel':
                    model_name, path = stmt[1], stmt[2].strip('"')
                    model = self.models.get(model_name)
                    if model is None:
                        raise ValueError(f"[inklang] No model named '{model_name}' to save")
                    save_model(model, path)
                elif stmt[0] == 'loadmodel':
                    model_name, path = stmt[1], stmt[2].strip('"')
                    model = load_model(path)
                    if model is None:
                        raise ValueError(f"[inklang] Failed to load model from '{path}'")
                    self.models[model_name] = model
                elif stmt[0] == 'if':
                    condition, stmts = stmt[1], stmt[2]
                    var, op, value = condition
                    var_value = self.variables.get(var, 0.0)
                    value = float(value)
                    if op == '>':
                        execute = var_value > value
                    elif op == '<':
                        execute = var_value < value
                    elif op == '==':
                        execute = var_value == value
                    elif op == '>=':
                        execute = var_value >= value
                    elif op == '<=':
                        execute = var_value <= value
                    elif op == '!=':
                        execute = var_value != value
                    if execute:
                        self.run_statements(stmts)
                elif stmt[0] == 'repeat':
                    count, stmts = int(stmt[1]), stmt[2]
                    for _ in range(count):
                        self.run_statements(stmts)
            except Exception as e:
                logging.error(f"Runtime error: {str(e)}")
                inkl_print(f"Runtime error: {str(e)}")

def transform_tree(node):
    if isinstance(node, lark.Tree):
        if node.data == 'start':
            return [transform_tree(child.children[0]) for child in node.children if child.children]
        elif node.data == 'print_stmt':
            return ('print', node.children[0].value)
        elif node.data == 'dataset_stmt':
            return ('dataset', node.children[0].value, node.children[1].value)
        elif node.data == 'train_stmt':
            return ('train', node.children[0].value, node.children[1].value, 
                    node.children[2].value if len(node.children) > 2 else None, 
                    node.children[-1].value)
        elif node.data == 'predict_stmt':
            return ('predict', node.children[0].value, node.children[1].value)
        elif node.data == 'savemodel_stmt':
            return ('savemodel', node.children[0].value, node.children[1].value)
        elif node.data == 'loadmodel_stmt':
            return ('loadmodel', node.children[0].value, node.children[1].value)
        elif node.data == 'if_stmt':
            condition = transform_tree(node.children[0])
            statements = [transform_tree(child.children[0]) for child in node.children[1:] if child.children]
            return ('if', condition, statements)
        elif node.data == 'repeat_stmt':
            count = node.children[0].value
            statements = [transform_tree(child.children[0]) for child in node.children[1:] if child.children]
            return ('repeat', count, statements)
        elif node.data == 'condition':
            logging.debug(f"Condition node children: {node.children}")
            if len(node.children) != 3:
                raise ValueError(f"[inklang] Invalid condition: expected 3 components (acc, CMP_OP, NUMBER), got {len(node.children)}: {node.children}")
            if node.children[0].type != 'ACC' or node.children[0].value != 'acc':
                raise ValueError(f"[inklang] Invalid condition: expected 'acc' as first token, got {node.children[0]}")
            return (node.children[0].value, node.children[1].value, node.children[2].value)
    return node

def parse_and_run(code):
    try:
        parser = lark.Lark(grammar, start='start', parser='earley', lexer='standard')
        tree = parser.parse(code)
        logging.debug(f"Parsed tree: {tree.pretty()}")
        statements = transform_tree(tree)
        runtime = InklangRuntime()
        runtime.run_statements(statements)
    except lark.exceptions.LarkError as e:
        logging.error(f"Parse error: {str(e)}")
        inkl_print(f"Parse error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        inkl_print(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inklang.py <script.inkl>")
        sys.exit(1)
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        code = f.read()
    logging.debug(f"Running script: {sys.argv[1]}")
    parse_and_run(code)