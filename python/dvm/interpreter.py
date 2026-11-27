import ast
import re
import numpy as np
import hashlib
import subprocess
import argparse


def read_input_file(input_filename):
    """
    读取输入文件内容。
    """
    with open(input_filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def extract_vgraph_blocks(content):
    """
    提取内容中的所有 vgraph 块。
    返回 vgraph_blocks 列表和每个 vgraph 块的起始位置列表。
    """
    vgraph_blocks = []
    block_positions = []
    for match in re.finditer(r'rgraph.eager\([^)]*\)\s*\{[^}]*\}', content, re.DOTALL):
        vgraph_blocks.append(match.group())
        block_positions.append(match.start())
    return vgraph_blocks, block_positions


def split_params(params):
    """
    自定义参数分割函数。
    在处理括号时进行正确的分割。
    """
    param_list = []
    current_param = ''
    bracket_level = 0
    for c in params:
        if c in '[<({':
            bracket_level += 1
        elif c in ']>)}':
            bracket_level -= 1
        if c == ',' and bracket_level == 0:
            param_list.append(current_param.strip())
            current_param = ''
        else:
            current_param += c
    if current_param:
        param_list.append(current_param.strip())
    return param_list


def parse_axis(axis_str):
    """
    安全解析 axis 参数，只接受 None、整数、整数列表或整数元组。
    """
    axis_str = axis_str.strip()
    if axis_str.lower() == 'none':
        return None

    try:
        axis_value = ast.literal_eval(axis_str)
    except (SyntaxError, ValueError) as err:
        raise ValueError(f"Invalid axis: {axis_str}") from err

    if type(axis_value) is int:
        return (axis_value,)

    if isinstance(axis_value, (list, tuple)):
        if all(type(dim) is int for dim in axis_value):
            return tuple(axis_value)

    raise ValueError(f"Invalid axis: {axis_str}")


def parse_variable_def(var_def, allow_extra=False):
    """
    解析变量定义字符串。
    匹配类似 '%0[1,2,3]<float32>' 的模式。
    返回 var_name, var_shape, var_type。
    如果 allow_extra 为 True，则仅提取变量名，忽略后面的形状和类型信息。
    """
    if allow_extra:
        var_pattern = r'^(%[\d]+)'
    else:
        var_pattern = r'^(%[\d]+)(?:\[([^\]]*)\])?(?:<([^>]+)>)?$'
    var_match = re.match(var_pattern, var_def)
    if var_match:
        var_name = var_match.group(1)
        if not allow_extra:
            var_shape_str = var_match.group(2)
            var_type = var_match.group(3)
            # 处理 bfloat16 类型
            if var_type == 'bfloat16':
                var_type = 'float16'
            # 解析形状
            if var_shape_str:
                var_shape = [int(s.strip()) for s in var_shape_str.split(',')]
            else:
                var_shape = None
        else:
            var_shape = None
            var_type = None
        return var_name, var_shape, var_type
    else:
        print(f"Error parsing variable definition: {var_def}")
        return None, None, None


def resolve_variable(var_name, skip_variables):
    """
    解析变量的最终映射，处理多级替换。
    """
    while var_name in skip_variables:
        var_name = skip_variables[var_name]
    return var_name


def parse_operation(operation_part):
    """
    解析操作部分，提取操作名、标量参数和参数列表。
    """
    op_match = re.match(r'^(\w+)(?:<([^>]*)>)?\((.*)\)$', operation_part)
    if op_match:
        operation = op_match.group(1)
        scalar_value = op_match.group(2)
        params = op_match.group(3).strip()
        params_list = split_params(params) if params else []
    else:
        operation = operation_part
        scalar_value = None
        params_list = []
    return operation, scalar_value, params_list


def process_vgraph_blocks(vgraph_blocks, vgraph_occurrences):
    """
    处理每个 vgraph 块：
    - 解析块
    - 生成代码
    - 统计算子数量
    返回：
    - vgraph_data_list: 包含每个唯一 vgraph 的相关数据的列表
    - total_operator_counts: 算子数量统计字典
    """
    # 存储每个唯一 vgraph 的数据，键为哈希值
    vgraph_data = {}
    # 总的算子数量统计，包括重复的 vgraph
    total_operator_counts = {}

    for idx, block in enumerate(vgraph_blocks):
        # 生成当前块的哈希值
        block_hash = hashlib.md5(block.encode('utf-8')).hexdigest()

        # 获取该 vgraph 块在文件中的出现总次数
        occurrence_count = vgraph_occurrences[block_hash]

        # 检查是否已处理过该 vgraph
        if block_hash in vgraph_data:
            # 已处理过，获取之前的算子数量
            operator_count = vgraph_data[block_hash]['operator_count']
            # 更新总的算子数量统计
            total_operator_counts[operator_count] = total_operator_counts.get(
                operator_count, 0) + 1
            continue  # 跳过重复的 vgraph

        # 解析并生成该 vgraph 块的代码
        code_lines, operator_count = parse_and_generate_code(
            block, idx, occurrence_count)

        # 保存当前 vgraph 的数据
        vgraph_data[block_hash] = {
            'code_lines': code_lines,
            'operator_count': operator_count,
            'occurrence_count': occurrence_count,
            'block': block,
            'idx': idx  # 原始顺序，用于在排序时保持稳定性
        }

        # 更新总的算子数量统计
        total_operator_counts[operator_count] = total_operator_counts.get(
            operator_count, 0) + 1

    # 将 vgraph_data 转换为列表，并按 occurrence_count 从大到小排序
    vgraph_data_list = list(vgraph_data.values())
    vgraph_data_list.sort(key=lambda x: (-x['occurrence_count'], x['idx']))

    return vgraph_data_list, total_operator_counts


def parse_and_generate_code(block, idx, occurrence_count):
    """
    解析单个 vgraph 块并生成测试代码。
    返回：
    - code_lines: 测试函数的代码行列表
    - operator_count: vgraph 中的算子数量
    """
    # 分割块为行
    lines = block.strip().split('\n')
    # 去除 vgraph 定义和结束的大括号
    lines = [line.strip() for line in lines]
    # 去掉 vgraph(...) { 和 }
    if lines[0].startswith('rgraph.eager'):
        lines = lines[1:-1]
    else:
        lines = lines

    # 初始化映射和列表
    variable_mapping = {}    # 映射 %0 到 x0
    variable_types = {}      # 存储每个变量的类型
    numpy_steps = {}         # 存储每个变量的 NumPy 计算步骤
    variable_shapes = {}     # 存储每个变量的 shape
    skip_variables = {}      # 标记需要跳过代码生成的变量
    code_lines = []          # 存储生成的代码行
    requires_positive_inputs = False  # 是否需要正数输入
    requires_integer_inputs = False   # 是否需要整数输入

    # 初始化算子计数器
    operator_count = 0

    # 添加注释，包含原始 vgraph 代码块和出现次数
    comment_lines = ['# 原始 vgraph 代码块：']
    for line in lines:
        comment_lines.append(f"# {line}")
    comment_lines.append(f"# 该代码块在文件中出现的总次数：{occurrence_count}")
    code_lines.extend(comment_lines)

    # 添加测试函数定义
    function_name = f"test_{idx}"
    code_lines.append(f"def {function_name}():")
    code_lines.append('    t = Tester("eager")')

    # **第一遍：收集变量信息，检查是否存在 Log 或 Sqrt 操作**
    for line in lines:
        if not line or line.startswith('//'):
            continue  # 跳过空行或注释

        if '=' not in line:
            continue
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        # 解析左侧
        var_name, var_shape, var_type = parse_variable_def(lhs)
        if not var_name:
            continue

        # 存储变量类型和形状
        variable_types[var_name] = var_type
        if var_shape:
            variable_shapes[var_name] = var_shape

        # 解析右侧
        if '//' in rhs:
            operation_part, comment_part = rhs.split('//', 1)
            operation_part = operation_part.strip()
            comment_part = comment_part.strip()
        else:
            operation_part = rhs.strip()
            comment_part = ''

        # 提取操作和参数
        operation, scalar_value, params_list = parse_operation(operation_part)

        # **检查是否存在 Log 或 Sqrt 操作**
        if operation in ['Log', 'Sqrt']:
            requires_positive_inputs = True
        if operation == 'Power':
            requires_integer_inputs = True

    # **第二遍：生成代码**
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith('//'):
            i += 1
            continue  # 跳过空行或注释

        # 分割左侧和右侧
        if '=' not in line:
            print(f"无效的赋值行: {line}")
            i += 1
            continue
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        # 解析左侧
        var_name, var_shape, var_type = parse_variable_def(lhs)
        if not var_name:
            i += 1
            continue

        # 解析右侧
        if '//' in rhs:
            operation_part, comment_part = rhs.split('//', 1)
            operation_part = operation_part.strip()
            comment_part = comment_part.strip()
        else:
            operation_part = rhs.strip()
            comment_part = ''

        # 提取操作和参数
        operation, scalar_value, params_list = parse_operation(operation_part)

        # 映射变量名到代码变量名（例如，%0 -> x0）
        code_var_name = 'x' + var_name[1:]
        variable_mapping[var_name] = code_var_name

        # 根据操作生成代码和 NumPy 计算步骤
        if operation == 'Load':
            # Load 操作不计入算子数量
            shape_ref = variable_shapes.get(var_name, (1,))
            if var_type.startswith('bool'):
                # 处理布尔类型
                line1 = f"{code_var_name}_a = np.random.choice([True, False], size={shape_ref})"
            elif var_type.startswith('int'):
                # 处理整数类型
                if requires_positive_inputs:
                    # 生成正整数，范围 [1, 100)
                    line1 = f"{code_var_name}_a = np.random.randint(1, 100, {shape_ref}, dtype=np.{var_type})"
                else:
                    # 生成包含正负整数，范围 [-100, 100)
                    line1 = f"{code_var_name}_a = np.random.randint(-100, 100, {shape_ref}, dtype=np.{var_type})"
            else:
                # 处理浮点类型
                if requires_positive_inputs:
                    line1 = f"{code_var_name}_a = np.abs(np.random.normal(0, 0.1, {shape_ref})).astype(np.{var_type}) + 1e-5"
                elif requires_integer_inputs:
                    # 生成整数值，存储为浮点数
                    line1 = f"{code_var_name}_a = np.random.randint(1, 10, {shape_ref}).astype(np.{var_type})"
                else:
                    line1 = f"{code_var_name}_a = np.random.normal(0, 0.1, {shape_ref}).astype(np.{var_type})"
            line2 = f"{code_var_name} = t.load({code_var_name}_a)"
            code_lines.append('    ' + line1)
            code_lines.append('    ' + line2)
            numpy_steps[var_name] = f"{code_var_name}_a"

        elif operation == 'Broadcast':
            # 处理 Broadcast 操作
            operator_count += 1  # 增加算子计数
            shape_ref = variable_shapes.get(var_name, [])
            param_var_name, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name = resolve_variable(
                param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            if param_code_var and param_numpy_expr:
                line = f"{code_var_name} = t.broadcast({param_code_var}, {shape_ref})"
                code_lines.append('    ' + line)
                numpy_expr = f"np.broadcast_to({param_numpy_expr}, {shape_ref})"
                numpy_steps[var_name] = numpy_expr
            else:
                print(f"Error: variable {param_var_name} not found.")
        elif operation in ['Add', 'Sub', 'Mul', 'Div', 'Power', 'Maximum', 'Minimum', 'Compare', 'LogicalOr', 'LogicalAnd']:
            if operation == "Compare":
                operation = "Less"
            if operation == "Power":
                operation = "Pow"
            operator_count += 1  # 增加算子计数
            if len(params_list) == 2:
                # 二元操作，两个参数
                param_var_name1, _, _ = parse_variable_def(
                    params_list[0], allow_extra=True)
                param_var_name2, _, _ = parse_variable_def(
                    params_list[1], allow_extra=True)
                param_var_name1 = resolve_variable(
                    param_var_name1, skip_variables)
                param_var_name2 = resolve_variable(
                    param_var_name2, skip_variables)
                param_code_var1 = variable_mapping.get(param_var_name1)
                param_code_var2 = variable_mapping.get(param_var_name2)
                param_numpy_expr1 = numpy_steps.get(param_var_name1)
                param_numpy_expr2 = numpy_steps.get(param_var_name2)
                if param_code_var1 and param_code_var2 and param_numpy_expr1 and param_numpy_expr2:
                    numpy_op_map = {
                        'Add': ('add', 'np.add'),
                        'Mul': ('mul', 'np.multiply'),
                        'Sub': ('sub', 'np.subtract'),
                        'Div': ('div', 'np.divide'),
                        'Pow': ('pow', 'np.power'),
                        'Maximum': ('maximum', 'np.maximum'),
                        'Minimum': ('minimum', 'np.minimum'),
                        'Less': ('less', 'np.less'),
                        'LogicalOr': ('logical_or', 'np.logical_or'),
                        'LogicalAnd': ('logical_and', 'np.logical_and'),
                    }
                    numpy_op = numpy_op_map.get(operation, 'np.add')
                    line = f"{code_var_name} = t.{numpy_op[0]}({param_code_var1}, {param_code_var2})"
                    code_lines.append('    ' + line)
                    numpy_expr = f"{numpy_op[1]}({param_numpy_expr1}, {param_numpy_expr2})"
                    numpy_steps[var_name] = numpy_expr
                else:
                    print(
                        f"Error: variable {param_var_name1} or {param_var_name2} not found.")
            elif len(params_list) == 1:
                operator_count += 1  # 增加算子计数
                # 二元操作，带标量参数
                param_var_def = params_list[0]
                param_var_name, _, _ = parse_variable_def(
                    param_var_def, allow_extra=True)
                param_var_name = resolve_variable(
                    param_var_name, skip_variables)
                param_code_var = variable_mapping.get(param_var_name)
                param_numpy_expr = numpy_steps.get(param_var_name)
                if scalar_value is None:
                    scalar_value = params_list[0]  # 如果没有 scalar_value，则参数可能是标量
                    param_code_var = None
                    param_numpy_expr = None
                if param_code_var and param_numpy_expr and scalar_value is not None:
                    line = f"{code_var_name} = t.binary('{operation}', {param_code_var}, {scalar_value})"
                    code_lines.append('    ' + line)
                    numpy_op_map = {
                        'Add': 'np.add',
                        'Mul': 'np.multiply',
                        'Maximum': 'np.maximum',
                        'Minimum': 'np.minimum',
                        'Div': 'np.divide',
                        'Less': 'np.less',
                    }
                    numpy_op = numpy_op_map.get(operation, 'np.add')
                    numpy_expr = f"{numpy_op}({param_numpy_expr}, {scalar_value})"
                    numpy_steps[var_name] = numpy_expr
                else:
                    print(f"Error: variable {param_var_name} not found.")
            else:
                print(
                    f"Unsupported operation: {operation} with parameters: {params_list}")
        elif operation == 'BroadcastS':
            # 处理 BroadcastS 操作
            operator_count += 1  # 增加算子计数
            if scalar_value is None:
                print(f"BroadcastS operation缺少标量值: {line}")
                i += 1
                continue
            # 获取目标形状
            shape_ref = variable_shapes.get(var_name, [])
            # 生成代码
            line = f"{code_var_name} = {scalar_value}"
            code_lines.append('    ' + line)
            numpy_expr = f"np.full({shape_ref}, {scalar_value}, dtype=np.{var_type})"
            numpy_steps[var_name] = numpy_expr
        elif operation == 'ScalarDiv':
            operator_count += 1
            param_var_name, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            line = f"{code_var_name} = t.binary(\"Div\", {scalar_value}, {param_code_var})"
            code_lines.append('    ' + line)
            numpy_expr = f"np.divide({scalar_value}, {param_numpy_expr})"
            numpy_steps[var_name] = numpy_expr
        elif operation in ['Abs', 'Exp', 'Tanh', 'Log', 'Sqrt', 'Reciprocal', 'IsFinite', 'LogicalNot']:
            operator_count += 1  # 增加算子计数
            if len(params_list) < 1:
                print(f"{operation} operation缺少参数: {line}")
                i += 1
                continue
            param_var_name, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            if param_code_var and param_numpy_expr:
                numpy_op_map = {
                    'Abs': ('abs', 'np.abs'),
                    'Exp': ('exp', 'np.exp'),
                    'Tanh': ('tanh', 'np.tanh'),
                    'Log': ('log', 'np.log'),
                    'Sqrt': ('sqrt', 'np.sqrt'),
                    'Reciprocal': ('reciprocal', 'np.reciprocal'),
                    'IsFinite': ('isfinite', 'np.isfinite'),
                    'LogicalNot': ('logical_not', 'np.logical_not'),
                }
                numpy_op = numpy_op_map.get(operation, 'np.abs')
                line = f"{code_var_name} = t.{numpy_op[0]}({param_code_var})"
                code_lines.append('    ' + line)
                numpy_expr = f"{numpy_op[1]}({param_numpy_expr})"
                numpy_steps[var_name] = numpy_expr
            else:
                print(f"Error: variable {param_var_name} not found.")
        elif operation == 'Cast':
            operator_count += 1  # 增加算子计数
            if len(params_list) < 1:
                print(f"Cast operation缺少参数: {line}")
                i += 1
                continue
            param_var_def = params_list[0]
            param_var_name, _, _ = parse_variable_def(
                param_var_def, allow_extra=True)
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            target_var_type = variable_types.get(var_name)
            if param_code_var and param_numpy_expr and target_var_type:
                line = f"{code_var_name} = t.cast({param_code_var}, \"{target_var_type}\")"
                code_lines.append('    ' + line)
                np_target_var_type = "bool_" if target_var_type == "bool" else target_var_type
                numpy_expr = f"{param_numpy_expr}.astype(np.{np_target_var_type})"
                numpy_steps[var_name] = numpy_expr
            else:
                print(
                    f"Error: variable {param_var_name} not found or target type missing.")
        elif operation == 'Copy':
            operator_count += 1  # 增加算子计数
            if len(params_list) < 1:
                print(f"Copy operation缺少参数: {line}")
                i += 1
                continue
            param_var_name, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            if param_code_var and param_numpy_expr:
                line = f"{code_var_name} = t.copy({param_code_var})"
                code_lines.append('    ' + line)
                numpy_expr = f"{param_numpy_expr}.copy()"  # 在 NumPy 侧新建变量
                numpy_steps[var_name] = numpy_expr
            else:
                print(f"Error: variable {param_var_name} not found.")
        elif operation == 'Reduce':
            operator_count += 1  # 增加算子计数
            if len(params_list) < 1:
                print(f"Reduce operation缺少参数: {line}")
                i += 1
                continue
            param_var_name, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            param_numpy_expr = numpy_steps.get(param_var_name)
            if param_code_var and param_numpy_expr:
                if scalar_value is not None:
                    scalar_params = split_params(scalar_value)
                    if len(scalar_params) == 2:
                        axis_str = scalar_params[0]
                        keepdims_str = scalar_params[1]
                        try:
                            axis = parse_axis(axis_str)
                        except ValueError as err:
                            print(err)
                            i += 1
                            continue
                        keepdims = True if keepdims_str.lower == 'true' else False
                    else:
                        print(f"Reduce not aixs")
                        i += 1
                        continue

                # 生成代码
                line = f"{code_var_name} = t.sum({param_code_var}, {axis}, {keepdims})"
                code_lines.append('    ' + line)
                if axis is None:
                    numpy_expr = f"np.sum({param_numpy_expr}, keepdims={keepdims})"
                else:
                    numpy_expr = f"np.sum({param_numpy_expr}, axis={axis}, keepdims={keepdims})"
                numpy_steps[var_name] = numpy_expr
            else:
                print(f"Error: variable {param_var_name} not found.")
        elif operation == 'MatMul':
            operator_count += 1
            trans_a, trans_b = scalar_value.split(", ")
            trans_a = "False" if trans_a == "0" else "True"
            trans_b = "False" if trans_b == "0" else "True"
            param_var_name1, _, _ = parse_variable_def(
                params_list[0], allow_extra=True)
            param_var_name2, _, _ = parse_variable_def(
                params_list[1], allow_extra=True)
            param_var_name1 = resolve_variable(
                param_var_name1, skip_variables)
            param_var_name2 = resolve_variable(
                param_var_name2, skip_variables)
            param_code_var1 = variable_mapping.get(param_var_name1)
            param_code_var2 = variable_mapping.get(param_var_name2)
            param_numpy_expr1 = numpy_steps.get(param_var_name1)
            param_numpy_expr2 = numpy_steps.get(param_var_name2)
            if trans_a == "True":
                param_numpy_expr1 = f"{param_numpy_expr1}.swapaxes(-1,-2)"
            if trans_b == "True":
                param_numpy_expr2 = f"{param_numpy_expr2}.swapaxes(-1,-2)"

            if len(params_list) == 2:
                line = f"{code_var_name} = t.matmul({param_code_var1}, {param_code_var2}, {trans_a}, {trans_b})"
                numpy_expr = f"np.matmul({param_numpy_expr1}.astype(np.float32), {param_numpy_expr2}.astype(np.float32)).astype(np.float16)"
            else:
                param_var_bias, _, _ = parse_variable_def(
                    params_list[2], allow_extra=True)
                param_var_bias = resolve_variable(
                    param_var_bias, skip_variables)
                param_code_bias = variable_mapping.get(param_var_bias)
                param_numpy_bias = numpy_steps.get(param_var_bias)
                line = f"{code_var_name} = t.matmul({param_code_var1}, {param_code_var2}, {trans_a}, {trans_b}, {param_code_bias})"
                numpy_expr = f"np.matmul({param_numpy_expr1}.astype(np.float32), {param_numpy_expr2}.astype(np.float32)).astype(np.float16) + {param_numpy_bias}"
            code_lines.append('    ' + line)
            numpy_steps[var_name] = numpy_expr
        elif operation == 'Store':
            # Store 操作不计入算子数量
            if params_list:
                param_var_name, _, _ = parse_variable_def(
                    params_list[0], allow_extra=True)
            else:
                param_var_name = var_name
            param_var_name = resolve_variable(param_var_name, skip_variables)
            param_code_var = variable_mapping.get(param_var_name)
            numpy_expr = numpy_steps.get(param_var_name)
            y_var_name = 'y' + code_var_name[1:]
            if param_code_var and numpy_expr:
                numpy_var_name = y_var_name + '_numpy'
                line_numpy = f"{numpy_var_name} = {numpy_expr}"
                code_lines.append('    ' + line_numpy)
                line = f"{y_var_name} = t.store_expect({param_code_var}, {numpy_var_name})"
                code_lines.append('    ' + line)
            else:
                print(f"Error: variable {param_var_name} not found.")
        else:
            print(f"Unsupported operation: {operation}")
        i += 1

    # 在函数末尾添加 assert t.run_check()
    code_lines.append('    assert t.run_check()')

    return code_lines, operator_count


def generate_test_code(vgraph_data_list, output_filename):
    """
    将生成的代码写入到输出文件。
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('import pytest\n')
        f.write('import numpy as np\n')
        f.write('from dvm.tester import Tester\n\n')
        # 写入所有测试函数
        for data in vgraph_data_list:
            for line in data['code_lines']:
                f.write(line + '\n')
            f.write('\n')  # 添加空行分隔不同的测试函数
    print(f"已将生成的代码写入到 {output_filename}")


def run_pytest(output_filename):
    """
    运行 pytest 并捕获输出。
    返回 pytest 输出的字符串。
    """
    process = subprocess.Popen(['pytest', output_filename],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True)
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # 实时打印 pytest 输出
        output_lines.append(line)
    process.wait()
    output_pytest = ''.join(output_lines)
    return output_pytest


def parse_pytest_output(output_pytest, output_filename):
    """
    解析 pytest 输出，找到失败的测试用例。
    返回失败的测试函数名称列表。
    """
    failing_tests = []
    for line in output_pytest.splitlines():
        match = re.match(
            r'FAILED\s+{}\:\:(test_\w+)'.format(re.escape(output_filename)), line)
        if match:
            failing_test = match.group(1)
            failing_tests.append(failing_test)
    return failing_tests


def extract_failing_tests(failing_tests, output_filename):
    """
    从生成的代码文件中提取失败的测试函数，并写入到 'test_wrong.py'。
    """
    if not failing_tests:
        print("所有测试均通过，无失败的测试用例。")
        return
    with open(output_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    failing_functions_code = []
    inside_failing_function = False
    indent_level = 0
    for i, line in enumerate(lines):
        if line.startswith('def test_'):
            function_name_match = re.match(r'def (test_\w+)\(\):', line)
            if function_name_match:
                function_name = function_name_match.group(1)
                if function_name in failing_tests:
                    inside_failing_function = True
                    indent_level = len(line) - len(line.lstrip())
                    # 添加前面的注释行
                    j = i - 1
                    while j >= 0 and lines[j].startswith('#'):
                        failing_functions_code.insert(0, lines[j])
                        j -= 1
                    failing_functions_code.append(line)
                else:
                    inside_failing_function = False
            else:
                inside_failing_function = False
        elif inside_failing_function:
            # 包含当前行
            failing_functions_code.append(line)
            # 检查函数是否结束
            if line.strip() == '':
                # 检查下一行的缩进
                if i + 1 < len(lines):
                    next_line_indent = len(
                        lines[i + 1]) - len(lines[i + 1].lstrip())
                    if next_line_indent <= indent_level:
                        inside_failing_function = False
        else:
            continue
    # 将失败的测试函数写入到 test_wrong.py
    with open('test_wrong.py', 'w', encoding='utf-8') as f:
        f.write('import pytest\n')
        f.write('import numpy as np\n')
        f.write('from dvm.tester import Tester\n\n')
        f.writelines(failing_functions_code)
    print("已将失败的测试用例写入到 test_wrong.py")


def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Process vgraph.eager() blocks and generate test code.')
    parser.add_argument(
        'input_filename', help='Input log file containing vgraph.eager() blocks')
    parser.add_argument('-o', '--output', default='test.py',
                        help='Output test code filename (default: test.py)')
    parser.add_argument('--no-run', action='store_true',
                        help='Do not run pytest after generating the test code')
    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output

    # 读取输入文件
    content = read_input_file(input_filename)

    # 提取 vgraph 块
    vgraph_blocks, block_positions = extract_vgraph_blocks(content)

    # 统计每个 vgraph 块的出现总次数
    vgraph_occurrences = {}
    for block in vgraph_blocks:
        block_hash = hashlib.md5(block.encode('utf-8')).hexdigest()
        vgraph_occurrences[block_hash] = vgraph_occurrences.get(
            block_hash, 0) + 1

    # 处理 vgraph 块，生成代码并统计算子数量
    vgraph_data_list, total_operator_counts = process_vgraph_blocks(
        vgraph_blocks, vgraph_occurrences)

    # 生成测试代码文件（按出现次数从大到小排序）
    generate_test_code(vgraph_data_list, output_filename)

    # 打印算子数量统计
    print("\n算子数量统计（除去 Load 和 Store）：")
    for count in sorted(total_operator_counts.keys()):
        num_vgraphs = total_operator_counts[count]
        print(f"包含 {count} 个算子的 vgraph 有 {num_vgraphs} 个")

    # 运行 pytest 并捕获输出
    # 根据命令行参数决定是否运行 pytest
    if not args.no_run:
        # 运行 pytest 并捕获输出
        output_pytest = run_pytest(output_filename)

        # 解析 pytest 输出，找到失败的测试用例
        failing_tests = parse_pytest_output(output_pytest, output_filename)

        # 提取失败的测试用例并写入到 test_wrong.py
        extract_failing_tests(failing_tests, output_filename)
    else:
        print("已生成测试代码，但未运行 pytest。")


if __name__ == '__main__':
    main()
