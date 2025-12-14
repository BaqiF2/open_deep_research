# Docstring 规范

所有 Python 代码的 docstring 必须使用 **Google 风格**。

## 格式要求

### 函数/方法

```python
def function_name(arg1: str, arg2: int) -> bool:
    """简短描述函数功能。

    更详细的描述（可选）。

    Args:
        arg1: 参数1的描述。
        arg2: 参数2的描述。

    Returns:
        返回值的描述。

    Raises:
        ValueError: 异常触发条件的描述。
    """
```

### 类

```python
class ClassName:
    """简短描述类的功能。

    更详细的描述（可选）。

    Attributes:
        attr1: 属性1的描述。
        attr2: 属性2的描述。
    """
```

### 模块

```python
"""模块的简短描述。

更详细的描述（可选）。

Example:
    使用示例::

        import module_name
        module_name.function()
"""
```

## 参考

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
